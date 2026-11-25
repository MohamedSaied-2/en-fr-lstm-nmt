import os
import time
import tensorflow as tf
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

from model import Encoder, Decoder

# --- 5 to 5.5 Hour Full-Dataset Hyperparameters ---
VOCAB_SIZE = 16000
MAX_LEN = 50            # Keeps ~96% of all 232,825 pairs (including long sentences)
EMBED_DIM = 384
LATENT_DIM = 512        # Full 512-unit LSTM memory state
DROPOUT = 0.2
EPOCHS = 10             # ~28-33 mins/epoch with bucketing (~5 to 5.5 hours total)
INITIAL_LR = 1e-3

ARTIFACTS_DIR = "./artifacts"
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]


def build_and_save_tokenizer(sentences, save_path: str, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", end_of_word_suffix="</w>"))
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
    )
    tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        end_of_word_suffix="</w>",
    )
    tokenizer.train_from_iterator(sentences, trainer=trainer)
    tokenizer.save(save_path)
    return tokenizer


def encode_split(split_data, en_tok: Tokenizer, fr_tok: Tokenizer, max_len: int):
    sos_id = fr_tok.token_to_id("<SOS>")
    eos_id = fr_tok.token_to_id("<EOS>")
    encoded = []

    for item in split_data:
        en_text = item["translation"]["en"].strip()
        fr_text = item["translation"]["fr"].strip()
        if not en_text or not fr_text:
            continue

        en_ids = en_tok.encode(en_text).ids
        fr_ids = fr_tok.encode(fr_text).ids
        if not en_ids or not fr_ids or len(en_ids) > max_len or len(fr_ids) > max_len - 1:
            continue

        # Reverse source tokens so the start of the English sentence sits closest to the decoder
        en_ids = en_ids[::-1]
        dec_in = [sos_id] + fr_ids
        dec_out = fr_ids + [eos_id]
        encoded.append((en_ids, dec_in, dec_out))

    return encoded


def create_tf_dataset(encoded_pairs, shuffle: bool = True):
    def gen():
        for en_ids, dec_in, dec_out in encoded_pairs:
            yield (en_ids, dec_in), dec_out

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=20000)

    # Group similar-length sentences together to eliminate wasted <PAD> math on CPU
    ds = ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.maximum(tf.shape(x[0])[0], tf.shape(x[1])[0]),
        bucket_boundaries=[12, 20, 30, 40],
        bucket_batch_sizes=[256, 256, 192, 128, 128],
        padded_shapes=(([None], [None]), [None]),
        padding_values=((0, 0), 0),
    )
    return ds.prefetch(tf.data.AUTOTUNE)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("1. Loading full IWSLT 2017 en-fr dataset...")
    dataset = load_dataset("IWSLT/iwslt2017", "iwslt2017-en-fr", trust_remote_code=True)
    train_raw = dataset["train"]
    val_raw = dataset["validation"]

    print(f"2. Training BPE tokenizers (vocab={VOCAB_SIZE}) on all {len(train_raw)} pairs...")
    en_tok = build_and_save_tokenizer(
        (x["translation"]["en"] for x in train_raw),
        os.path.join(ARTIFACTS_DIR, "en_tokenizer.json"),
        VOCAB_SIZE,
    )
    fr_tok = build_and_save_tokenizer(
        (x["translation"]["fr"] for x in train_raw),
        os.path.join(ARTIFACTS_DIR, "fr_tokenizer.json"),
        VOCAB_SIZE,
    )
    pad_id = fr_tok.token_to_id("<PAD>")

    print("3. Encoding and bucketing sentence pairs...")
    train_pairs = encode_split(train_raw, en_tok, fr_tok, MAX_LEN)
    val_pairs = encode_split(val_raw, en_tok, fr_tok, MAX_LEN)
    print(f"   Kept {len(train_pairs)} train pairs and {len(val_pairs)} validation pairs.")

    train_ds = create_tf_dataset(train_pairs, shuffle=True)
    val_ds = create_tf_dataset(val_pairs, shuffle=False)

    print("4. Initializing 512-unit LSTM Encoder-Decoder...")
    encoder = Encoder(en_tok.get_vocab_size(), EMBED_DIM, LATENT_DIM, DROPOUT)
    decoder = Decoder(fr_tok.get_vocab_size(), EMBED_DIM, LATENT_DIM, DROPOUT)

    # Build Keras 3 variables cleanly before training/checkpointing
    dummy_x = tf.zeros((1, 1), dtype=tf.int32)
    dummy_h, dummy_c = encoder(dummy_x, training=False)
    decoder(dummy_x, initial_state=[dummy_h, dummy_c], training=False)

    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR, clipnorm=1.0)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    def masked_loss(y_true, logits):
        loss = loss_fn(y_true, logits)
        mask = tf.cast(tf.not_equal(y_true, pad_id), dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(mask), 1.0)

    @tf.function
    def train_step(enc_in, dec_in, dec_target):
        with tf.GradientTape() as tape:
            state_h, state_c = encoder(enc_in, training=True)
            logits, _, _ = decoder(
                dec_in, initial_state=[state_h, state_c], training=True
            )
            loss = masked_loss(dec_target, logits)
        variables = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))
        return loss

    @tf.function
    def val_step(enc_in, dec_in, dec_target):
        state_h, state_c = encoder(enc_in, training=False)
        logits, _, _ = decoder(
            dec_in, initial_state=[state_h, state_c], training=False
        )
        return masked_loss(dec_target, logits)

    ckpt = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
    manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("5. Starting full-dataset training...")
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        total_train_loss, train_batches = 0.0, 0

        for step, ((enc_in, dec_in), dec_target) in enumerate(train_ds, start=1):
            batch_loss = train_step(enc_in, dec_in, dec_target)
            total_train_loss += float(batch_loss)
            train_batches += 1
            if step % 250 == 0:
                print(
                    f"Epoch {epoch} | Batch {step} | Train Loss: {total_train_loss / train_batches:.4f}"
                )

        total_val_loss, val_batches = 0.0, 0
        for (enc_in, dec_in), dec_target in val_ds:
            total_val_loss += float(val_step(enc_in, dec_in, dec_target))
            val_batches += 1

        avg_train_loss = total_train_loss / max(train_batches, 1)
        avg_val_loss = total_val_loss / max(val_batches, 1)
        elapsed = time.time() - start_time
        current_lr = float(optimizer.learning_rate.numpy())

        print(
            f"--> Epoch {epoch}/{EPOCHS} ({elapsed:.1f}s) | LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            saved_path = manager.save()
            print(f"    Saved new best checkpoint: {saved_path}")
        else:
            epochs_without_improvement += 1
            new_lr = max(current_lr * 0.5, 1e-5)
            optimizer.learning_rate.assign(new_lr)
            print(f"    Val loss did not improve. Reduced LR to {new_lr:.6f}")
            if epochs_without_improvement >= 3:
                print("    Early stopping triggered (3 epochs without improvement).")
                break


if __name__ == "__main__":
    main()