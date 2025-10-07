= Fully connected stimulus reconstruction neural network with a single hidden layer
![[nonlinear1.png]]

**input**: LC neurons (L=number of time lags, C=number of EEG channels)
**hidden layer**: 2 neurons with tanh-activation
**output**: 1 neuron that uses linear activation and outputs 1 sample of the reconstructed envelope

**Trainable parameters:** 2 × (LC + 1) (hidden layer) +2 + 1 (output layer) ≈ 3446 trainable parameters

**cost function**: network is trained to minimize 1 − ρ(sˆa, sa) over a segment of M training samples (with ρ() the pearson correlation)