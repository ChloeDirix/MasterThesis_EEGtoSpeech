= minimum mean squared error between the actual attended envelope and the reconstructed envelope. This is equivalent to the Pearson correlation coefficient.

--> most basic way of training the decoder

<span style="background:rgba(3, 135, 102, 0.2)">least square regression: </span>
![[MMSE.png]]
X= EEG design matrix
s<sub>a</sub> is the attended envelope


<span style="background:rgba(3, 135, 102, 0.2)">Solution</span>

![[formula_d.png]]
X<sup>T</sup> X= autocorrelation matrix of EEG
X<sup>T</sup>s<sub>a</sub>= cross-correlation between EEG and attended speech

<span style="background:rgba(3, 135, 102, 0.2)">How to avoid overfitting?</span>

Regularization:
- **Ridge regression (L2 regularization):** Adds a penalty λ∥d∥² to shrink weights smoothly.

- **Lasso regression (L1 regularization):** Adds a penalty λ∥d∥ to force many weights to be zero → sparse decoder

Lasso is solved with iterative algorithms like ADMM. Ridge is straightforward (closed-form solution)

<span style="background:rgba(3, 135, 102, 0.2)">Training across multiple segments</span>
Two ways (for K segments)

- **Late integration:** Train a decoder dk​ for each segment separately, then average the decoders.

- **Early integration:** Concatenate all data and train one decoder directly.

--> mathematically equivalent if weights are averaged properly, but different in practice

![[linearMethods.png]]
