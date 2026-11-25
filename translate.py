import os
import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

from model import Encoder, Decoder

# Must match train.py hyperparameters
EMBED_DIM = 384
LATENT_DIM = 512
MAX_LEN = 50

ARTIFACTS_DIR = "./artifacts"
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")


class Translator:
    def __init__(self):
        self.en_tok = Tokenizer.from_file(
            os.path.join(ARTIFACTS_DIR, "en_tokenizer.json")
        )
        self.fr_tok = Tokenizer.from_file(
            os.path.join(ARTIFACTS_DIR, "fr_tokenizer.json")
        )

        self.sos_id = self.fr_tok.token_to_id("<SOS>")
        self.eos_id = self.fr_tok.token_to_id("<EOS>")
        self.pad_id = self.fr_tok.token_to_id("<PAD>")

        self.encoder = Encoder(self.en_tok.get_vocab_size(), EMBED_DIM, LATENT_DIM)
        self.decoder = Decoder(self.fr_tok.get_vocab_size(), EMBED_DIM, LATENT_DIM)

        # Build Keras 3 variables before restoring checkpoint
        dummy_x = tf.zeros((1, 1), dtype=tf.int32)
        dummy_h, dummy_c = self.encoder(dummy_x, training=False)
        self.decoder(dummy_x, initial_state=[dummy_h, dummy_c], training=False)

        ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
        latest_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
        if not latest_ckpt:
            raise FileNotFoundError(
                f"No checkpoint found in {CHECKPOINT_DIR}. Run train.py first."
            )
        ckpt.restore(latest_ckpt).expect_partial()
        print(f"Restored weights from: {latest_ckpt}\n")

    def _encode_source(self, sentence: str):
        en_ids = self.en_tok.encode(sentence.strip()).ids[:MAX_LEN][::-1]
        enc_in = tf.constant([en_ids], dtype=tf.int32)
        return self.encoder(enc_in, training=False)

    def greedy_translate(self, sentence: str) -> str:
        state_h, state_c = self._encode_source(sentence)
        next_token = tf.constant([[self.sos_id]], dtype=tf.int32)
        output_ids = []

        for _ in range(MAX_LEN):
            logits, state_h, state_c = self.decoder(
                next_token, initial_state=[state_h, state_c], training=False
            )
            pred_id = int(tf.argmax(logits[0, -1, :]).numpy())
            if pred_id == self.eos_id:
                break
            if pred_id != self.pad_id:
                output_ids.append(pred_id)
            next_token = tf.constant([[pred_id]], dtype=tf.int32)

        return self.fr_tok.decode(output_ids)

    def beam_search_translate(
        self, sentence: str, beam_width: int = 5, length_penalty: float = 0.7
    ) -> str:
        state_h, state_c = self._encode_source(sentence)
        # Each beam: (token_id_list, cumulative_log_prob, state_h, state_c)
        beams = [([self.sos_id], 0.0, state_h, state_c)]

        for _ in range(MAX_LEN):
            if all(seq[-1] == self.eos_id for seq, _, _, _ in beams):
                break

            candidates = []
            for seq, score, h, c in beams:
                if seq[-1] == self.eos_id:
                    candidates.append((seq, score, h, c))
                    continue

                dec_in = tf.constant([[seq[-1]]], dtype=tf.int32)
                logits, next_h, next_c = self.decoder(
                    dec_in, initial_state=[h, c], training=False
                )
                log_probs = tf.nn.log_softmax(logits[0, -1, :]).numpy()
                top_k_ids = np.argsort(log_probs)[-beam_width:]

                for tok_id in top_k_ids:
                    tok_id = int(tok_id)
                    candidates.append(
                        (seq + [tok_id], score + float(log_probs[tok_id]), next_h, next_c)
                    )

            candidates.sort(
                key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True
            )
            beams = candidates[:beam_width]

        best_seq = beams[0][0]
        clean_ids = [
            tid for tid in best_seq if tid not in (self.sos_id, self.eos_id, self.pad_id)
        ]
        return self.fr_tok.decode(clean_ids)


if __name__ == "__main__":
    translator = Translator()
    sample_sentences = [
        "Thank you so much.",
        "How are you doing today?",
        "I work as a doctor, and my brother works as an engineer.",
        "This is a very important problem.",
        "We need to find a better solution for the future.",
    ]

    for text in sample_sentences:
        greedy_fr = translator.greedy_translate(text)
        beam_fr = translator.beam_search_translate(text, beam_width=5)
        print(f"EN    : {text}")
        print(f"Greedy: {greedy_fr}")
        print(f"Beam  : {beam_fr}")
        print("-" * 50)