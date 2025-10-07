= AAD algorithm is<font color="#76923c"> training-free </font>and <font color="#76923c">adaptive</font> in time (non-stationary)
- <span style="background:rgba(136, 49, 204, 0.2)">no pre training needed</span> --> avoid big training datasets
- <span style="background:rgba(136, 49, 204, 0.2)">Adaptive</span> --> adjusts to non-stationary EEG changes in real time
- <span style="background:rgba(136, 49, 204, 0.2)">sparse solutions</span> --> easier to interpret, less overfitting

<u>Mathematically</u>
![[Lasso.png]]
	- si,l= speech envelope _i_ in window _l_.
    - Xl= EEG data matrix for that window
    - d = decoder weights to be estimated.
    - First term ∥si,l−Xl d∥ = reconstruction error (MMSE part)
    - Second term λq ∥d∥​ = **L1 regularization** (lasso) --> encourages sparsity

<u>Attention decoding</u>
for each window:
	- two decoders are estimated
	- Attended decoder → stronger, sparser peaks (higher L1 norm)

<u>Remarks</u>
- **EEG channel selection**: To avoid overfitting, only a subset of channels is used    
- **Regularization parameter (λq​)** is chosen via cross-validation.