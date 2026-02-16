# -*- coding: utf-8 -*-

# import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def reverse_complement(seq):
    complement = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
    return ''.join(complement[base] for base in reversed(seq))

def shift_sequence(seq, shift=1, fill='N'):
    # shift >0: right shift, <0: left shift
    if shift > 0:
        return fill*shift + seq[:-shift]
    elif shift < 0:
        return seq[-shift:] + fill*(-shift)
    else:
        return seq

def main():
    with open("promoters.data", "r") as f:
        lines = f.readlines()

    sequences = []
    labels = []

    for line in lines:
        parts = line.strip().split(",")

        label = 1 if parts[0] == "+" else 0
        seq = parts[2].strip().upper()

        # create more dataset
        # Augmentation: shifts and reverse complement
        for s in [-1, 0, 1]:
            sequences.append(shift_sequence(seq, shift=s))
            sequences.append(reverse_complement(seq))
            labels.append(label)
            labels.append(label)

    dna_dict = {'A':0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}

    def encode(seq):
        return [dna_dict[base] for base in seq]

    X = np.array([encode(seq) for seq in sequences])
    y = np.array(labels)

    print("Input shape:", X.shape)
    print("Label shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=25, stratify=y
    )

    class TransformerModel(layers.Layer):
        def __init__(self, embed_dim, ff_dim, rate=0.01):
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
            attn_output = self.att(inputs, inputs) # residuals
            attn_output = self.dropout1(attn_output, training=training) #prevent overfitting
            out1 = self.layernorm1(inputs + attn_output) # stabilize learning

            ffn_output = self.ffn(out1) # adds non linear again
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)

    maxlen = len(sequences[0])
    vocab_size = len(list(dna_dict))
    embed_dim = 16
    ff_dim = 32

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
    x = embedding_layer(inputs)

    # Positional encoding
    positions = tf.range(start=0, limit=maxlen, delta=1)
    pos_embedding = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
    x = x + pos_embedding(positions)

    transformer_block = TransformerModel(embed_dim, ff_dim)
    x = transformer_block(x, training=None)

    # Summarizes the whole DNA sequence into one representative feature vector for classification
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    # final classifier
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer = "adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X_train, y_train, epochs=10, batch_size=25)

    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)

    preds = model.predict(X_test)
    preds = np.argmax(preds, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    main()
