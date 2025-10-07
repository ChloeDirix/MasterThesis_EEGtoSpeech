##### <u>Backward modeling/ decoding</u>
Reconstruct the speech envelope from the EEG channels using a neural
= **Stimulus reconstruction approach**

		EEG -(1)-> Speech Envelope <-(2)-> speakers


- (1) = the neural decoder
	Differences between algorithms: The way the neural decoder is pre-trained to optimally reconstruct the speech envelope, while blocking other neural activity
	
- (2) = Correlation step: 
	The reconstructed speech envelope is correlated with the speech envelopes of all speakers. Using the <font color="#76923c">Pearson correlation</font> over window of  τ second = decision window length	--> effect on accuracy
	Highest Pearson correlation --> attended speaker

##### <u>Forward modeling/ Encoding: Using an encoder</u>
Estimate the neural response in each EEG channel, based on the speech envelopes

	Attended Speech -(1)-> neural response (reconstructed EEG) <-(2)-> EEG

- (1) = encoder
- (2) = correlation


> *"backward MISO decoding models have been demonstrated to outperform forward encoding models as the former can exploit the spatial coherence across the different EEG channels at its input. (Geirnaert et al., 2021, p. 4)*

##### <u>Direct classification</u>
The attention is directly predicted in an end-to-end fashion, without explicitly reconstructing the speech envelope.

--> usually supervised methods: require ground truth 

- **subject-specific fashion:** based on EEG data from the actual subject under test
- **subject-independent fashion**: based on EEG data from other subjects than the subject under test

| Comparison          |                                                                                                                     |
| ------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Subject specific    | Higher accuracy                                                                                                     |
| Subject independent | Leads to a universal decoder. Can be applied to any subject without the need to go through the ground-truth process |


##### Example of the different types (linear and non linear)
| Approach                  | Output                     | Linear methods                                            | Non-linear methods                                                 |
| ------------------------- | -------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------ |
| **Forward (encoding)**    | Neural response prediction | Linear Temporal Response Function (TRF), ridge regression | Deep neural networks (CNNs, RNNs, Transformers), kernel regression |
| **Backward (decoding)**   | Stimulus reconstruction    | Linear reconstruction decoder                             | Nonlinear regression (DNNd, autoencoders)                          |
| **Direct classification** | Attention label (A vs B)   | LDA, logistic regression, linear SVM                      | Kernel SVM, RF, CNN, RNN, Transformers, contrastive learning       |
