compare C × T EEG segment with a 1 × T speech envelope directly using CNN

![[nonlinear2.png]]
**output**: similarity score ∈ \[0, 1] using binary cross-entropy cost function
**CNN**: 2 convolutional layers + 4 fully connected layers (FC)
	ELU=exponential linear unit -> used as nonlinear activation function

**trainable parameters**: 64 × (C + 1) × L1 (first convolutional layer) +2 × 64 × L2 (second convolutional layer) +200 × 3 (first fully connected layer) +200 × 201 (second FC layer) +100 × 201 (third FC layer) +101 (fourth FC layer) ≈ 69070 trainable parameters