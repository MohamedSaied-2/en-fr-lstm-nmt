import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size: int, embed_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        self.lstm = tf.keras.layers.LSTM(
            latent_dim, dropout=dropout, return_state=True
        )

    def call(self, x, training=False):
        x = self.embedding(x)
        _, state_h, state_c = self.lstm(x, training=training)
        return state_h, state_c


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size: int, embed_dim: int, latent_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim, mask_zero=True
        )
        self.lstm = tf.keras.layers.LSTM(
            latent_dim, dropout=dropout, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, x, initial_state, training=False):
        x = self.embedding(x)
        outputs, state_h, state_c = self.lstm(
            x, initial_state=initial_state, training=training
        )
        logits = self.dense(outputs)
        return logits, state_h, state_c