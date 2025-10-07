## Background
Recent studies use natural speech during EEG 

Challenges?
<font color="#205867">-</font> <span style="background:rgba(3, 135, 102, 0.2)">Requires strict time alignment</span>
	  *(e.g., auditory brainstem responses require ~600 ms precision)*
	  
<font color="#205867">-</font> <span style="background:rgba(3, 135, 102, 0.2)">Data collection is expensive </span> 
	  - Personal data acquisition is expensive  
	  - need for public datasets!
	  
<font color="#205867">-</font> <span style="background:rgba(3, 135, 102, 0.2)">Limitations of current public datasets</span> 
	  - Too small in scale: <font color="#c00000">Machine learning algorithms need a lot of data!</font>
	  - Difficult to combine different dataset due to differences in: signal acquisition equipment, measurement protocols, preprocessing

---

## This Dataset
Total: 169 hours of EEG data
participants: 18-30 years, normal hearing, Dutch native and no neurological pathologies affecting brain responses

![EEG dataset illustration](dataset%20info.png)


---

## Methods
<font color="#4f6128">Trial</font> = uninterrupted recording of 15 min

<font color="#4f6128">session</font> = complete set of trials and pre-screening activities that a participant underwent from the moment they entered until the moment they left

<font color="#4f6128">Stimulus</font> = The speech audio files presented to the participants, designed to elicit specific responses from the brains
![[session.png]]

##### Behavioral experiments
- air conduction tresholds: hearing treshold test with frequencies and intensities
- Flemish matrix test: speech reception treshold (**SRT**): measures how well someone understands speech in noise 
---

##### EEG data collection
###### Setup
- triple-walled, soundproof booth equipped with a Faraday cage to reduce external electromagnetic interference
- minimize muscle movements
- appropriate head cap by measuring head size
- All 84 offsets were ideally between +20 and -20 mV
- sampling rate of 8192 Hz
- All participants listened to 6, 7, 8, 10 trials of each 15 min with breaks

###### Stimulus categories

| Stimulus types        |                           |
| --------------------- | ------------------------- |
| reference audiobook   | made for children         |
| audiobooks            | made for adults/ Children |
| audiobooks with noise | SNR of 5 dB               |
| Podcasts              | scientific of nature      |
| Podcasts with video   | extra video               |

###### Session types

| Session Type        | Participants | Stimuli | Special Conditions |
|---------------------|--------------|---------|--------------------|
| **ses-shortstories01** | 26 | 10 audiobook parts (children’s stories) | • 2/10 trials in noise (SNR = 5 dB)<br>• 3 participants: pitch-shifted versions of audiobook_1<br>• 1 artefact control (audiobook_1 without insertphones) |
| **ses-varyingstories** | 59 | Reference audiobook_1 + 1 split audiobook (~30 min) + 3–5 podcasts | • Balanced male/female speakers<br>• Stimuli rotated every 2–8 participants |

![[stimuli.png]]

---

##### Stimulus preparation
make sure the sounds are precisely controlled and synchronized with EEG recording

1. time-stamps
	<span style="background:rgba(3, 135, 102, 0.2)">Stimuli</span>: stored at 48kHz sampling
	<span style="background:rgba(3, 135, 102, 0.2)">trigger file</span>: for each stimulus a trigger is send to the EEG system, creating time-marks in the signal

2. noise creation 
	 white noise filterzs it so its spectrum matches that of speech, but without intelligibility
	 
3. Calibration
	make sure audio playback has consistent loudness across participants:

---

##### Krios data
= 3D scan of the EEG caps for all participants

---

##### EEG data preprocessing
dataset contains raw + preprocessed EEG (https://github.com/exporl/auditory143)

Preprocessing
1. HP filter: 1st-order Butterworth filter with a cut-off frequency of 0.5 Hz. 
2. Zero-phase filtering: forward and backward. 
3. Downsampling: from 8192 Hz to 1024 Hz 
4. Eyeblink artifact removal -- multichannel Wiener filter
5. Re-referenced to a common average
6. Downsampled to 64 Hz

##### Stimuli preprocessing
1. original: 48kHz
2. calculate envelope using gammatone filterbank with 28 subbands
		Subband: take absolute value of each sample, raised to power of 0.6
3. Average subband to have 1 envelope
4. Downsampling to 64 Hz

---

## Dataset Structure
The repository contains **3 main parts**:

1. **Raw Data**
   - Folder per participant (sub-xxx, where xxx = 001–085), contains sessions: ses-shortstories01 or ses-varyingstoriesxx (xx = 01–09)  
   
   - <u>Each session has</u>:
     - **beh/** → behavioral results (SRT)  
	        *Files named with participant, session, task = listeningActive, and run (1–3)*  
     - **eeg/** → EEG recordings, with files per trial:  
       1. `*_eeg.bdf.gz` – Raw EEG (BioSemi, 8192 Hz)  
       2. `*_eeg.apr` – Extra experiment info (incl. question answers)  
       3. `*_stimulation.tsv` – Links EEG with stimulus timing  
       4. `*_events.tsv` – Stimulus presentation timing details  
     - **Tasks**:  
       - *listeningActive* (stimulus playback)  
       - *restingState* (silence, start/end)  

2. **Stimuli**  
   - For each stimulus, 4 files: 
     1. Stimulus audio (`stimulusName.npz.gz`, 48 kHz)  
     2. Noise file (`noise_stimulusName.npz.gz`)  
     3. Trigger file (`t_stimulusName.npz.gz`)  
     4. Experiment description (`stimulusName.apx`)    

3. **Preprocessed Data**  
   - Structured similarly to raw data (per participant, per session).  
   - Key differences:  
     - Downsampled for easier analysis  
     - File naming based on raw EEG source file  
     - Extra suffix: `desc-preproc` (prevents name conflicts)  
     - Includes **stimulus name** to directly link EEG response to stimulus.  

---

## Technical Validation of dataset quality
https://github.com/exporl/auditory-eeg-dataset
-->experiments on preprocessed version of the dataset

1. Split each trial (for each participant) into training (80%), validation(10%) and test set(10%)
2. Normalise each train, test and validation with train mean and std

##### 1. Linear Forward/Backward Models

*Backward model*: Predict stimulus envelope from EEG → test for neural tracking  

*Forward model*: Predict EEG from stimulus envelope → visualize spatial EEG response patterns.  

###### Training 
- **Integration window**: -100 ms to +400 ms.  
- **Regularization**: Ridge regression, λ chosen via leave-one-out cross-validation from range 10^[-6…6]
- **Evaluation**: Pearson correlation between predicted and true signals. 
	→ Neural tracking confirmed if score > 95th percentile of null distribution (created by 100 random circular shifts)

###### Results
- **Neural tracking** detected for all but 11 of 666 recordings  
- **Delta band** gave highest performance
- **Stimulus comparison** (delta-band):  
	  - Audiobooks: median correlation **0.184**  
	  - Podcasts: median correlation **0.133**  
	  - Significant difference (Mann–Whitney U, p < 10⁻⁹)
![[linear decoder performance.png]]
	
- **Forward model topomaps**: strongest correlations in **temporal** and **occipital** EEG channels 
![[forward model performance.png]]


---

##### 2. Non-linear Model: Match–Mismatch Paradigm
Model sees EEG + matching stimulus segment + mismatched stimulus segment (imposter)

**Imposter** = segment shifted by 1s

<u>Architecture:</u> Dilated convolutional neural network.  
  1. Reduce EEG channels (64 → 8) with 1D convolutional layer
  2. Apply N dilated conv layers (shared for EEG & stimulus)
  3. Compute cosine similarity between EEG & both stimulus envelopes.  
  4. Feed similarity scores into sigmoid neuron → predict correct match

**Training**:  
  - Implemented in TensorFlow, Adam optimizer (learning rate lr=0.001).  
  - Loss function: binary cross-entropy.  
  - Early stopping (patience = 5, max 50 epochs).  
  - Input length = 5s for training; tested with 1, 2, 3, 5, 10s segments.  
  
**Results**:  
  - Accuracy increases with longer segment length
  - Generalization holds when mismatched segment chosen randomly (not just +1s shift)

---