# promoter-transformer

This project builds a small CNN and Transformer model to classify short DNA sequences as promoter or non-promoter regions using the UCI Molecular Biology dataset. 

For the transformer, DNA sequences are tokenized into numerical representations and passed through an embedding layer. A lightweight Transformer encoder learns sequence-wide patterns through self-attention, allowing it to detect biologically meaningful motifs such as promoter consensus regions. The final sequence representation is pooled and fed into a classifier to predict promoter probability  

For the CNN, DNA sequences are encoded to pytorch tensors where each nucleotide is represented as a 5-dimensional vector. These tensors are passed through 1D convolutional layers that act as motif detectors, learning local sequence patterns such as promoter consensus regions. The convolutional layers extract important local features, which are then flattened and passed through fully connected layers. A final sigmoid activation outputs the probability that a sequence is a promoter. The model is trained using binary cross-entropy loss and optimized with Adam.