### <font color="#4f6128">1. Introduction</font>

**This study?**
- new dataset, collected from listeners with a hearing impairment (HI).  --> more representative of real-world challenges
- prove a non linear model with contrastive learning is better. Three models were implemented:
	- Baseline linear model **LM**
	- non LM without contrastive learning **NLM**
	- non LM with contrastive learning **NLMwCL**


**Contrastive learning?**
= a technique for training machine learning models that transforms inputs into representation vectors, mapping similar inputs closer and dissimilar inputs farther apart based on a distance function, like cosine similarity

<font color="#31859b">SigLIP</font>= multimodal contrastive learning technique based on CLIP 
[[SigLip]]

<u>comparison</u>

| Model                | (dis)advantages                                                                                                                                                                                                                                                                |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LM                   | **+** simple<br>**-** don't capture complex, nonlinear dynamics of brain activity<br>                                                                                                                                                                                          |
| NLM                  | **+** promising results in many fields of brain signal processing (emotion recognition, ...)<br>**-** large amount of data needed<br>**-** intersubject variability making generalisation difficult<br>**-** limitation in interpretability<br>**-** computationally expensive |
| Contrastive learning | **+** learn discriminative features that can differentiate attended and unattended speech<br>**+** handles limitation of data<br>**+** comes close to core challenge of differentiating speech (simple positive and negative pairs)                                            |

---

### <font color="#4f6128">2. Materials and methods</font>

#### 1. Participants
- N = 34 native Danish speakers (age: 21–84, mean = 64.2, SD = 13.6). After preprocessing 31 remained
- mild-to-moderately severe symmetrical sensorineural hearing loss
#### 2. Experimental Design
- 84 trials per participant:
    - 4 familiarization + 80 test.
    - Each trial: 5 s babble → 33 s speech.
        
- Noise reduction (NR) conditions:
    - **System 1**: 16-channel MVDR + Wiener filter.
    - **System 2**: 24-channel MVDR + DNN-based postfilter.
    - Each tested with <font color="#92d050">NR ON / NR OFF</font> 
    
#### 3. Preprocessing

- EEG:
    - Filters: 0.5–70 Hz bandpass, 49–51 Hz notch.
    - Downsampling: 1024 → 256 Hz → 64 Hz.
    - Artifact rejection: manual channel removal + exclusion of 3 participants.
    - Final trial size: (64 channels × 2112 samples).    
- Speech features:
    - Envelope: Hilbert transform → 64 Hz.
    - 32-band mel-spectrogram (FFT, 512-sample windows, 250 overlap).
    - Features concatenated → trial size: (33 features × 2112 samples).

#### 4. Dataset Splitting
 Avoiding leakage: Inter-trial split (not intra-trial)
 Partitioning: 80% train, 10% validation, 10% test.
 Balanced across NR conditions (ON/OFF × system).

| Condition              | Train | Validation | Test |
| ---------------------- | ----- | ---------- | ---- |
| NR ON (DNN)            | 496   | 62         | 62   |
| NR OFF (DNN)           | 496   | 62         | 62   |
| NR ON (Wiener filter)  | 496   | 62         | 62   |
| NR OFF (Wiener filter) | 496   | 62         | 62   |
| **Total**              | 1984  | 248        | 248  |

---

### <font color="#4f6128">3. Making the models</font>

<u>linear model: </u>
--> linear regression model
estimate a decoder so the speech envelope can be reconstructed and compared to the attended and unattended speaker.

<u>Contrastive learning framework</u>
![[sridhar1.png]]

##### **Goal:** 
1 Reconstruct the speech envelope 
2 Classify attention 

To avoid overfitting on small datasets, they use a convolutional model with attention layers and skip connections instead of a heavy transformer
##### **EEG Encoder**
Input: raw EEG signals.
output: EEG embedding
![[sridhar2.png]]

1. **subject-specific layer** 
	→ learns separate weights per subject to account for individual differences
2. combine subject specific info with original
3. **Convolutional layer** to get the desired embedding size
4. **K-block** is repeated K times (K = hyperparameter) to extract spatial features from EEG
		<span style="background:rgba(240, 107, 5, 0.2)">Convolution</span>→<span style="background:rgba(240, 107, 5, 0.2)"> Multi-head attention (3 heads)</span>→<span style="background:rgba(240, 107, 5, 0.2)"> Dropout</span>→<span style="background:rgba(240, 107, 5, 0.2)">Normalization</span>

##### **Speech Encoder**
![[sridhar3.png]]

Input: speech envelope.
output: embedded audio
--> similar to EEG encoder

##### **Speech reconstruction module**
![[sridhar4.png]]
input: EEG embedding 
output: reconstructed speech envelope.
       

##### **Contrastive learning (SigLIP loss)**

S=speech data (NxFxT)
R=EEG data (NxCxT)

1.  S and R are passed through encoders and projected into the same space (NxVxT)
		Ŝ= speech embeddings 
		R̂=EEG embeddings
		
2. Compute a similarity matrix: each EEG sample compared with each speech sample in the batch.
		**diagonal** = positive pairs
		**off-diagonals** = negative pairs
		
3. Loss function encourages matched EEG–speech embeddings to be close, unmatched to be far.

##### **training**

Total loss = combination of:

1. **Contrastive loss (SigLIP)** → ensures embeddings align across modalities.
2. **Pearson correlation loss** → ensures reconstructed speech envelope matches the true envelope.
    
---

### <font color="#4f6128">4. Results</font>
![[sridhar5.png]] 
Classification accuracy: attended vs. ignored speech
Reconstruction accuracy: Pearson correlation between actual and reconstructed speech

- shaded region = std from the mean
- All models have increased accuracy and reconstruction accuracy with increased window size. 
- NLMwCL outperforms everything

**per subject performance**
![[sridhar6.png]]
- NLMwCL outperforms everything
- performance varies across individuals

---
### <font color="#4f6128">5. Discussion</font>
- proposed model outperforms other models
- models were specifically analysed for a 3-second window
- validation accuracy peaked before the validation loss triggered the early stopping, resulting in a temporary decrease in accuracy over several epochs, while the loss continued to decrease
- there may be better parameters possible
- high variability between participants
- study participants had HIs, which led to different EEG patterns then in other studies. But on different data it was also performing well
- Accuracy is always highly dependent on the amount of data
- The model architecture could be made more complex in the future