###### Sources:
*Linear Modeling of Neurophysiological Responses to Speech and Other Continuous Stimuli: Methodological Considerations for Applied Research - Crosse et al

*Relating EEG to continuous speech using deep neural networks: a review - Puffay et al*

###### EEG
1) **High-pass filter**: 
	    **goal**: remove unwanted DC shifts or slow drift potentials caused by electrode junction potentials (eg sweat)
	    .
	    *Recommended:* 
		    - cuttoff <1 Hz (usually 0.5 Hz)
		    - zero-phase shift filters? but not causality is a problem?
	    
2) **LP filter:**
		**Goal**: remove any unwanted high frequency noise that may be present, for example, due to muscle contractions or environmental interference such as 50/60-Hz line noise 	--> improve model performance
	.	
		**remark**:  - if high frequency needed: notch filter or zapline
				- non linear studies can benefit from using higher frequencies
	.
		*Recommended*: 
			- zero-phase filters
			- range \[2, 40] Hz
			
	
3) **Band-pass filter** 
		Often in linear studies: 2-9 Hz or 1-8 Hz (delta–theta band)
	    --> It has been shown that speech envelope and EEG recordings correlate best within the δ and θ band frequencies. 
	     
4) **Re-reference** to average or mastoids.
	    - To 1 channel (mastoid): enhance neural activity in a region of interest
	    - To mean of all channels: by subtracting the mean over all channels from each individual channel --> increase SNR
		
5) **Downsampling**
		**Range**: ~64–128 Hz
		**Goal**: reduce computational time during training. Often combined with anti-aliasing filter or a simple LP filter (below Nyquist frequency)
    
6) **Artifact removal** 
		ICA= independent component analysis
		MWF= multichannel wiener filtering
    
7) **Remove first 500-1000 ms of data**
		Avoid fitting the model to the neural response elicited by the onset of stimulation, as this is often a higher magnitude response
		
8) optional: **Normalizing**: important when using subject-independent techniques. all data within the same subject should be normalized together, not separately
		--> only if completely necessary

*extra step*: more specific features from EEG, such as a latent representation optimized through the training of an AE (Bollens et al 2022), or source-spatial feature images (SSFIs) (Tian and Ma 2020)

###### Speech
1) **Speech envelope** extraction (Hilbert or gammatone) or mel spectrogram,...
    
2) **Align EEG and envelope**
