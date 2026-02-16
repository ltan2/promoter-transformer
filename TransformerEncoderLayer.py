from tensorflow.keras import layers
from tensorflow import keras

class TransformerEncoderLayer(layers.Layer):
    def __init__(self, embed_dim, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(
            num_heads=2,
            key_dim=embed_dim
        )
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6) # after att
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6) # after ffn
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output, attn_scores = self.att(inputs, inputs, return_attention_scores=True) # residuals
        attn_output = self.dropout1(attn_output, training=training) #prevent overfitting
        out1 = self.layernorm1(inputs + attn_output) # stabilize learning

        ffn_output = self.ffn(out1) # adds non linear again
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_scores