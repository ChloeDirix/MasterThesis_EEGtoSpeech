Determine the spatial locus of attention (direction of attention) directly based on EEG using CNN

<font color="#9bbb59">+</font> does not require individual speech envelopes 
<font color="#9bbb59">+</font> no correlation coefficient needed over large windows --> avoids delays
![[nonlinear3.png]]
**input**: C × T EEG segment
**CNN**: 1 convolutional layer and 2 fully connected layers
**trainable parameters**: 5 × C × L (convolutional layer) +5 × 6 (first FC layer) +2 × 6 (second FC layer) ≈ 2708

**cost function**: cross-entropy cost function using mini-batch gradient descent
**regularization**(avoid overfitting): decay regularization + including other subject 