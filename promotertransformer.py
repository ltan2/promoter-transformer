# -*- coding: utf-8 -*-

# import libraries
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from ModelDef import ModelDef
import tensorflow as tf
import numpy as np
import functions

def main():
    sequences, labels = functions.load_data('promoters.data')

    X = np.array([functions.encode(seq) for seq in sequences])
    y = np.array(labels)

    print("Input shape:", X.shape)
    print("Label shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=25,stratify=y
    )

    model_obj = ModelDef()
    model = model_obj.transformer_def(sequences)
    
    model.compile(
        optimizer = "adam",
        loss=[
            "sparse_categorical_crossentropy",  # for classification
            None                                # no loss for attention
        ],
        metrics=["accuracy", None]
    )

    model.fit(X_train, [y_train, np.zeros((len(y_train),))], epochs=20, batch_size=35)

    loss, acc = model.evaluate(X_test, y_test)
    
    functions.evaluate_results(model, acc, X_test, y_test)

    attention_model = model_obj.attention_def(model)
    attention_weights = attention_model.predict(X_test)
    print(attention_weights.shape)
    functions.plot_attention(attention_weights)


if __name__ == "__main__":
    main()
