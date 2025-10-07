linear models are limited as they assume linearity in the EEG-speech relationship
**But**: brain has nonlinear dynamics

**Goal of this paper**:
- address methodology and pitfalls
- summarize main DL studies


## <font color="#31859b">1. Introduction</font>

**Why relate speech to brain processes?**
	<font color="#c00000">1)</font> understanding neural mechanisms
	<font color="#c00000">2)</font> measure processes in the brain for diagnostic of hearing
	<font color="#c00000">3)</font> designing auditory prosthesis

**Current literature proposes 2 approaches:**
- <u>Assuming single sound source</u>
	=> quantify the time-locking of the brain response to 1 stimulus = <span style="background:rgba(3, 135, 102, 0.2)">neural tracking</span>

	Applications: <font color="#c00000">1)</font>, <font color="#c00000">2)</font>

| Method                              | Main idea                                                                                                                                           | Note?                                                                                                                                 |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| MM                                  | Train model to associate EEG to corresponding  speech.                                                                                              | [[Notes Bollens et al. --- Contrastive Representation Learning with Transformers for Robust Auditory EEG Decoding#**Generic model**]] |
| Direct regression of stimulus (R/P) | Stimulus feature is reconstructed from EEG (or inv) and correlated to original signal (forward, backward [[Types of AAD algorithms]] , CCA [[CCA]]) | [[[Notes Geirnaert et al.--- EEG-based Auditory Attention Decoding#The different algorithms]]]                                        |
Accuracy = measure for neural tracking
	
- <u>Assuming multiple sound sources</u>
	=> determine which speech source was attended by the listener = <span style="background:rgba(3, 135, 102, 0.2)">AAD</span>

	application: <font color="#c00000">3)</font>, <font color="#c00000">1)</font>
	

| Methods                               | Description                                                                        | Advantage                                                                             |
| ------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| SI: speaker identity                  | Model is trained to which speech stream is attended given EEG and N speech streams | /                                                                                     |
| DFA: Directional focus classification | Decodes directional focus of attention (DFA)                                       | **-** avoids separate speech sources<br>**-** Possibility to use brain lateralization |
Accuracy = measure for auditory attention


## <font color="#31859b"> 2. DL techniques</font>

| Models |                                 |
| ------ | ------------------------------- |
| GRNNs  | general regression NN           |
| FCNN   | fully connected NN              |
| CNN    | convolutional NN                |
| LSTM   | long-short term memory          |
| GRU    | gated recurrent unit            |
| GANs   | generative adversarial networks |
| AEs    | autoencoders                    |
#### 2.1. Preprocessing?

EEG
	1) HP: remove any unwanted DC shifts or slow drift potentials
	2) LP: remove high frequencies to improve SNR
	3) Downsampling to reduce computational time during training (in accordance with previous filtering to avoid alisasing) (typically 128/ 64 Hz)
	4) artefact removal: multichannel Wiener or ICA
Speech
	**features used?** --> speech envelope, Mel spectrogram, fundamental frequency of the voice, phonemes, higher level speech features
	**Preprocessing?** HP filtering can lead to edge artefact and normalisation can lead to leaked information between training and test set 


#### 2.2. Data Segmentation

- <span style="background:rgba(173, 239, 239, 0.55)">Training set</span>: train the model by adjusting the weights and biases to minimize a loss function
- <span style="background:rgba(173, 239, 239, 0.55)">Validation set:</span> Tune hyperparameters during the training process
- <span style="background:rgba(173, 239, 239, 0.55)">Test set:</span> unseen data used to test the final performance without any bias

cross-validation: method that iterates over different data segmentations to train and validate a given model

#### 2.3. Evaluation metric

**single source paradigms**: classification metrics (MM, subject classification accuracy, sentence class acc)

**R/P**: Pearson correlation, MSE, MCD

**multi-speaker**: attention decoding accuracy


## <font color="#31859b">3. Recommendations</font>

| problem                                                              | reason                                                                                                                                                                                                                                                                                                                     | implication                                                                                                |
| -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| overfitting to trials, subjects or datasets --> bad generalizability | too small datasets                                                                                                                                                                                                                                                                                                         | results of studies are overly optimistic                                                                   |
| Correlation                                                          | **-** EEG and speech signals are not independent: they’re autocorrelated and filtered, which means even unrelated segments can look correlated "by chance". Standard correlation significance tests don’t account for this<br><br>**-** Correlation depends on many factors: preprocessing, conditions, architecture, .... | **-** correlation values are misleading<br><br>**-** Correlations are difficult to compare between studies |
|                                                                      |                                                                                                                                                                                                                                                                                                                            |                                                                                                            |

#### 3.1 Training, validation & test set

<u>Balancing the dataset</u>
	In Two competing speaker scenario
	measurement usually spread out in trials
	--> To generate a balanced dataset every trial the subjects have to pay attention to another speaker

<u>Split in training/validation/test?</u>
	common method: split each individual trial into training, validation and test 
	-> <span style="background:rgba(163, 67, 31, 0.2)">info leaking</span>! Always use held out trials for the test set

<u>Evaluation?</u>
use multiple datasets that differ in experimental conditions + test generalizability


#### 3.2 Interpretation of correlation scores

common method to compare predicted signal with EEG/speech envelope and evaluate the model. 
<span style="background:rgba(163, 67, 31, 0.2)">=> Not independent signals</span> ==> null distribution
<span style="background:rgba(240, 107, 5, 0.2)">=> not comparable between studies</span> ==> Always report multiple metrics to allow comparison

<u>Null distribution</u>
To show model really learned something: compare to **Null distribution**

How to create?
- <font color="#e36c09">permutation test</font>: shuffle alignment by circularly shifting --> preserves data structure, but destroys relationship
	- if model’s correlation > 95th percentile of null distribution **= statistically significant**
	- Problem: can create discontinuities, multiple permutations needed to get stable permutations
- <font color="#e36c09"> Phase scrambling</font>: randomize phase, keep frequency structure
	- problem: biased result

#### 3.3 Subject specific training vs subject independent

Subject-specific
**+** can be finetuned to a persons unique brain patterns, noise, ....
**+** often achieves higher accuracy for that specific person as it doesn't have to generalize
**-** training data from every new subject: time-consuming and impractical 

subject-independent
**+** no need to re-train for each subject
*+* large, diverse datasets can be used
**-** performance is lower 
**-** Good generalisation is difficult

**Recommendation:**
	Subject independent + finetuning to boost performance


#### 3.4 Negative samples selection in MM tasks

<u>Training phase: suggestions to avoid learning spurious cues with the MM task
</u>
Negative samples must be 'hard negatives' = challenging. That way, the model must learn **subtle brain–speech alignment**, not simple differences in distribution

Tips
- Pick mismatched speech segments that are **close in time** to the matched ones
- each speech segment should be labeled once as matched and once as mismatched
		Example:
			- Segment A + EEG → "match"
		    - Segment A + unrelated EEG → "mismatch"
		    
- sample mismatched segments from the same speech stimulus as the matched segments to avoid speaker differences 

<u>Evaluation phase: recommendations for a robust measure</u>

- **dm** = correspondence measure for the **match** pair  
    (EEG + correct speech segment)
- **dmm** = correspondence measure for the **mismatch** pair  
    (EEG + wrong speech segment)

| methods                                                                                                                                                       | advantages?                                                                                                                                                 |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I) For each matched candidate, compute _all possible_ mismatched scores (all dmm) and then compare their average to dm                                        | **+** limits variability of the decision criterion (due to averaging)<br>**-** computationally expensive                                                    |
| II) For each matched candidate, pick one arbitrary mismatched segment and compare that dmm to dm                                                              | **+** simple and comp cheap<br>**-** more variability then (I) and (III) as temporal proximity may vary between mismatched segments                         |
| III) For all matched candidates, pick a mismatched segment at a fixed delay and compare dmm to dm. Then check that the results are consistent with Method II. | **+** less time-comsuming then (I)<br>**+** less variability as (II)<br>**-** does not ensure the generalisation with other shifts --> bias in performance? |


