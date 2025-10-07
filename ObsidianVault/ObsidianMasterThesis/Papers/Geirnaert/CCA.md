= instead of only building a decoder, also build an **encoder** (forward model) from speech 


<u>what it does?</u>
input: **x(t)**: EEG data 
	 **s<sub>a</sub>(t):** Speech envelope

- **Decoder**: **<font color="#76923c">Backward model:</font>** 
		EEG → speech envelope
		linear filter wx
- **Encoder**: <font color="#76923c">Forward model:</font> 
		speech → EEG
		linear filter: wsa
- **CCA optimization**: fit both models jointly so the EEG and speech are maximally correlated. Solved via generalized eigenvalue decomposition
![[CCA.png]]
		- Numerator = cross-correlation between filtered EEG and filtered speech
	    - Denominator = normalization so it’s a correlation coefficient (Pearson).

<u>Extension to multiple independent dimensions</u>
Compute J forward models and J backward models 
 - EEG components are uncorrelated 
 - speech components are uncorrelated

**for each speaker i** --> vector of correlation coefficients between backward and forward model
ρi=\[ ρ<sub>i,1</sub>, ρ<sub>i,2</sub>,…,ρ<sub>i,J</sub> ]<sup>T</sup>

<font color="#76923c">→ richer feature set, better classification</font>

<u>Classification step</u>
To decide which is attended speaker
	f= ρ<sub>1</sub> - ρ<sub>2</sub> = feature vector for classification
	.
	LDA (linear discriminant analysis) classifier uses all J features, but gives more weight to the more informative components
	.
	PCA processing reduces dimensionality (kind of regularization)