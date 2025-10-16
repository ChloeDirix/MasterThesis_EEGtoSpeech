
## **Intro**
Understanding neural mechanisms of auditory processing by decoding of continuous speech from electroencephalography (EEG) is needed for applications in hearing diagnostics

## EEG? 
- + **Noninvasive** and **macroscopic** brain recording method
- + high temporal resolution: ideal for studying _time-varying neural responses_ to _continuous, natural stimuli_ (e.g., speech
- + ease of use
- - Main challenge: how to model and analyze such complex, time-dependent brain responses.

==applications:== 
- auditory neuroscience research
- diagnostic of hearing loss
- understand neural underpinnings of speech processing => develop objective measures for hearing diagnostics

---

## **Relate EEG to continuous speech?**

- linear regression: 
		+ good for speech intelligibility
		- problems due to low signal to noise ratio
- *ANNs* = artificial neural networks
		+ includes speech denoising
		+ includes signal enhancement
		+ predicting EEG from accoustic features
		+ better decoding performance
		+ improves accuracy (in match mismatch task)
		
### **Challenges**
- low speech to noise ratio in the recorded signals: <span style="color:rgb(75, 145, 127)"><b>AANs</b></span>
- no standardized public auditory EEG dataset + variations in evaluation metrics hinders comparison between different models: <span style="color:rgb(75, 145, 127)"><b>self-supervised learning techniques</b> --> <b>contrastive learning</b></span>

---

## **Contrastive learning**
<span style="font-weight:bold; color:rgb(52, 72, 111)">= a self supervised learning technique</span>

works by contrasting examples of similarity and dissimilarity in auditory stimuli. Leading to improved decoding performance and meaningful representations

	+ impressive results in multiple domains: 
		- natural speech processing
		- computer vision
		- EEG decoding
		- bridging modalities (eg CLIP)
		- brain imaging
	+ potential to relate complex, disparate data
	

---

## **New model architecture**
*contrastive learning + transformer networks [[Transformer network]]*
	
--> capture relationships between auditory stimuli and EEG responses
![[ModelArchitecture.png]]
### 1. EEG Encoder
Repeated N times (340000 parameters per repetition)

	Input = EEG time series (T time steps x C channels)
	Token = feature vector of size d_model
	output = sequence of tokens over time (length T)

[[EncoderSteps]]


### 2. Speech module
makes speech features look like EEG features so they can be compared. (both go through convolution, normalization, GELU, and skip connections).

1. <b>fully connected layer</b>
	projection so both EEG and speech have the same “feature dimension”
	The raw speech input has multiple channels/features, that need to be projected/recombined into 64 channel
2. <b>convolution layer</b> (same as before)
3. <b>Alignment with EEG</b>
	model temporal dependencies
	--> two bidirectional LSTM layers of dimension 64 and 4 respectively


### 3. Contrastive loss
teaches the model to match the EEG to the correct speech and separate it from wrong speech

		Given at time i:
		- EEG segment Ei
		- speech segment Si
					
#### <span style="color:rgb(52, 72, 111)">Background match-mismatch loss:</span>
- For each Ei - one correct Si
- The model learns to distinguish the correct one from one incorrect (negative) stimulus.
	
#### <span style="color:rgb(52, 72, 111)"><i>generalisation: contrastive loss</i></span>
- for each Ei - One correct Si and K candidates
- model must identify the correct match
- This forces the model to learn discriminative features that separate the true pair from all others.

[[Contrastive Loss Steps]]


---

## **Generic model**
= model works across participants without needing to be re-trained or adapted to each person.

#### **1. Averaging**
- **Dealing with randomness in initialization**? (random weights)
		train 16 models and report the average accuracy per subject
-  **Enhancing the prediction accuracy**?
		ensemble model approach:  averages the logits (the unnormalized output scores) from multiple models

#### **2. Evaluation**
Two tasks from the 2023 ICASSP Auditory EEG decoding challenge
![[GenericModel.png]]

<u>match-mismatch classification</u> 
= classification task based on the concept of matched/mismatched pairs (EEG, Speech).
.
	***training***
		The model gets three inputs (3s)
			- EEG segment
			- matched speech segment
			- mismatched speech segment
			.
		<span style="font-weight:bold; color:rgb(75, 145, 127)">Goal</span>: determine which of the two input stimulus segments correspond to the EEG segment
		<span style="font-weight:bold; color:rgb(75, 145, 127)">Remark</span>: make training more challenging by defining the mismatch stimulus close to the matched stimulus
		.
	***Testing***
		= first or second half of each EEG recording and half of the recordings
		.
		1. Calculate embeddings Ei, S1 & S2
		2. similarity
		3. matched = max similarity 
		4. calculate mean accuracy per subject 
	.	
	Result ==87%==

<u>stimulus envelope regression</u> 
= reconstructing the EEG speech envelope, then calculating the Pearson correlation = measure for similarity
.
	1. stimuli are split into segments of 60 seconds, each segment gets an envelope
			- split into segments of 3s with 50% overlap
			- calculate embeddings
			- simple linear model (TensorFlow with a kernel size of 32)
			- predict the envelope
	2. calculate mean correlation value per subject and over all subjects for test sets

Result ==0.176==

---

## **Finetuning**
linear models: subject-specific models often used approaches
for neural networks: limited data --> overfitting

solution? --> <span style="font-weight:bold; color:rgb(236, 81, 112)">Finetuning</span>
<span style="color:rgb(236, 81, 112)">= starting from a general model and adapting it to each subject</span>


---

## **How many subjects are needed?**
high enough to aid generalization
low enough to prevent overfitting

	1 subject: accuracy of 53%
	13 subjects: acc of 70%
	50 subjects: acc of 80%
	after: strong diminishing returns
	final score: 84%

#### **Ablation Study**
systematically modifying specific aspects of the model architecture and evaluating the resulting impact on accuracy

- **EEG Path**: Removing or simplifying the transformer => reduced accuracy. Adding more transformer layers =>  improved performance
    
- **Speech Path**: A smaller speech path: reduced performance slightly. Replacing **mel spectrograms** with **wav2vec (layer 19)** improved accuracy, significantly so for held-out subjects in task 1 (p<0.01), but had no impact on regression correlations.
		But mel spectrogram have lower dimensionality (better)
    
- **Ensembling**: Using an ensemble of wav2vec 19 models yielded the largest improvements, significantly boosting both accuracy and correlation in tasks 1 and 2 (p<0.001).

--> The EEG transformer depth and ensemble modeling were critical for strong performance, while speech path modifications had smaller effects.

--- 

## **Preprocessing**
#### **- EEG**
1. downsampling 8192 Hz -> 1024 Hz
2. HP filter (Butterworth 0.5 Hz)
3. zero-phase filtering
4. eyeblink artefact removal (wiener filter)
5. re-referencing to common average
6. downsampling to 64 Hz
#### **- Audio**
extract different speech representations:
-  Estimate speech envelope: used for regression task
- Mel-Spectrogram: capture perceptually relevant spectral features
- Wav2vec: capture phonetic/linguistic info.

---

## **DataSet**

| Data                |                                                                                                                                                                                |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| EEG recording setup | ~200 young adults (18–45), Dutch/Flemish native speakers with normal hearing. Data collected in a shielded, sound-proof booth with high-quality EEG equipment (64 electrodes). |
| Public data         | About half of the dataset (105 participants) is publicly available and used in the ICASSP EEG decoding challenge → makes results comparable across studies.                    |

**Training:** mix of the challenge dataset + extra subjects.
**Testing:** two setups to test generalization:
1. <span style="font-weight:bold; color:rgb(236, 81, 112)">Held-out stories</span>: same people, new material
2. <span style="font-weight:bold; color:rgb(236, 81, 112)">Held-out subjects</span>: new people, same setup
    
→ This checks if models can generalize to both new speech and new listeners.


#### **Training Details**
.
	**Framework:** <span style="font-weight:bold; color:rgb(75, 145, 127)">PyTorch </span>→ flexible deep learning environment.
    **Optimization:**  <span style="font-weight:bold; color:rgb(75, 145, 127)">Adam optimizer</span> with weight decay + learning rate scheduling 
	    → gives the most stable and reliable training.
    **Model structure:**  10 attention blocks,  8 heads 
	    → allows the model to capture complex patterns in the EEG–speech relationship.
    **Data handling:**    
	    - EEG + audio: split into 3-second windows 
		    → keeps training manageable and consistent.
	    - Batches (128 samples): contained both matching (positive) and mismatched (negative) EEG–audio pairs → teaches the model to distinguish real brain–speech alignment.
    **Shuffling across subjects:** after each epoch, EEG segments were remixed → prevents overfitting and improves generalization.
	**Normalization:** EEG recording normalized (mean = 0, variance = 1) 
		→ ensures signals are comparable across participants.
	**Statistical tests:**  Wilcoxon test + Holm-Bonferroni correction 
		→ ensures significance claims are rigorous and account for multiple comparisons.

---
	
### **Baseline Methods**
--> provide reference points to judge whether the new model truly improves over existing approaches by comparing against both _classic models_ and _top challenge winners_.

1. **Dilated Convolutional Network:**
    - Separate pathways for EEG and audio.
    - Uses dilated convolutions to capture temporal patterns at multiple scales.
    - Matches EEG–audio pairs via cosine similarity.
    - **Why:** A strong but relatively simple deep learning baseline for EEG–speech alignment.
        
2. **Thornton et al. (2023) – Challenge winner:**
    - Ensemble of 50 dilated convolutional models (majority voting).
    - Adds extra speech features (high-frequency modulations + f0).
    - Finetunes per subject for better accuracy.
    - **Why:** Demonstrates state-of-the-art performance by combining ensembling, richer features, and personalization.
        
3. **Linear Decoder:**
    - Simple 1D convolution over EEG (≈500 ms window).
    - Trained across all subjects without adaptation.
    - **Why:** Provides a very basic baseline to show how much more advanced models improve.
        
4. **Piao et al. (2023) – Top regression model:**
    - Transformer-based architecture with subject-specific conditioning.
    - Achieved large performance boost (≈13% better correlations).
    - Limitation: cannot apply subject-conditioning to unseen participants.
    - **Why:** Shows how modern transformer methods with personalization push performance further.
        
