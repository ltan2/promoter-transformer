from tensorflow.keras import layers
from tensorflow import keras
from TransformerEncoderLayer import TransformerEncoderLayer
import tensorflow as tf

class TransformerDef:

    def transformer_def(self, sequences):
        dna_dict = {'A':0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        maxlen = len(sequences[0])
        vocab_size = len(list(dna_dict))
        embed_dim = 32
        ff_dim = 64

        inputs = layers.Input(shape=(maxlen,))
        embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        x = embedding_layer(inputs)

        # Positional encoding
        positions = tf.range(start=0, limit=maxlen, delta=1)
        pos_embedding = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        x = x + pos_embedding(positions)

        encoder_layer1 = TransformerEncoderLayer(embed_dim, ff_dim)
        encoder_layer2 = TransformerEncoderLayer(embed_dim, ff_dim)

        x, attn1 = encoder_layer1(x, training=None)
        x, attn2 = encoder_layer2(x, training=None)

        # Summarizes the whole DNA sequence into one representative feature vector for classification
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        # final classifier
        outputs = layers.Dense(2, activation="softmax")(x)

        model = keras.Model(inputs=inputs, outputs=[outputs, attn2])

        return model
    
    def attention_def(self, model):
        attention_model = keras.Model(
            inputs=model.input,
            outputs=[model.output[1]] # attention scores
        )

        return attention_model
