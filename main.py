# -*- coding: utf-8 -*-

# import libraries
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from ModelDef import TransformerDef
from CNNModel import DNA_CNN
import tensorflow as tf
import numpy as np
import functions
import torch
import torch.nn as nn
import torch.nn.functional as F

def main():
    sequences, labels = functions.load_data('promoters.data')

    X = np.array([functions.encode(seq) for seq in sequences])
    y = np.array(labels)

    print("Input shape:", X.shape)
    print("Label shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=25,stratify=y
    )

    X_train_cnn = F.one_hot(torch.tensor(X_train, dtype=torch.long), num_classes=5).float()
    X_test_cnn  = F.one_hot(torch.tensor(X_test, dtype=torch.long), num_classes=5).float()

    y_train_cnn = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test_cnn  = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)


    transformer_obj = TransformerDef()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model = DNA_CNN().to(device)
    transformer_model = transformer_obj.transformer_def(sequences)
    
    transformer_model.compile(
        optimizer = "adam",
        loss=[
            "sparse_categorical_crossentropy",  # for classification
            None                                # no loss for attention
        ],
        metrics=["accuracy", None]
    )

    transformer_model.fit(X_train, [y_train, np.zeros((len(y_train),))], epochs=20, batch_size=35)
    batches_cnn = functions.train_loader(X_train_cnn, y_train_cnn)
    cnn_model.fit(batches_cnn, device)

    loss, acc = transformer_model.evaluate(X_test, y_test)
    print("Test accuracy for transformer: ", acc)
    
    functions.evaluate_results(transformer_model, acc, X_test, y_test)
    cnn_model.eval()
    functions.cnn_acc(cnn_model,X_test_cnn,y_test_cnn)

    attention_model = transformer_obj.attention_def(transformer_model)
    attention_weights = attention_model.predict(X_test)
    print(attention_weights.shape)
    functions.plot_attention(attention_weights)


if __name__ == "__main__":
    main()
