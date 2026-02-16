from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    
def load_data(file_name):
    with open(file_name, "r") as f:
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
    return sequences, labels

def encode(seq):
    dna_dict = {'A':0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return [dna_dict[base] for base in seq]

def evaluate_results(model, acc, X_test, y_test):
    print("Test Accuracy:", acc)

    preds = model.predict(X_test)[0]
    preds = np.argmax(preds, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

def plot_attention(attention, head=0):
    L = attention.shape[-1]
    uniform = 1.0 / L
    delta = attention[0, head] - uniform

    sns.heatmap(delta, cmap="coolwarm", center=0)
    plt.title(f"Attention Head {head} (Deviation from Uniform)")
    plt.show()

    # plt.figure(figsize=(8, 6))
    # sns.heatmap(attention[0, head], cmap="viridis")
    # plt.xlabel("Key Position")
    # plt.ylabel("Query Position")
    # plt.title(f"Attention Head {head}")
    # plt.show()
