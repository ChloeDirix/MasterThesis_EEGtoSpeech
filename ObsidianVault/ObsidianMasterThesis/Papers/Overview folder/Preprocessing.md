# <font color="#205867">Preprocessing</font>

## <font color="#974806">Combine dataset</font>
### **Possible challenges**
![[image-7.png]]
![[image-9.png]]

### **Extra: might be useful/needed?**

- **Adapters** = extra module between input and model, that handles specific dataset differences. 
- **Track dataset ID** as metadata for each sample. Makes data splits easier
- **DANN?**: feature based adaptation method that learns a domain-invariant representation through adversarial training.

<font color="#9bbb59">--> DANN and adapters might go too far, also no idea if implementable in this specific case??</font>

---

## <font color="#974806">EEG preprocessing</font>
<span style="background:rgba(163, 67, 31, 0.2)">minimize hand-crafted steps to let the model learn more</span>
### **1. Band-Pass**
**Goal**: capture speech envelope frequencies.
**HP filter:** remove unwanted DC shifts or slow drift potentials caused by electrode junction potentials (eg sweat)
**LP filter**: remove any unwanted high frequency noise that may be present, for example, due to muscle contractions or environmental interference such as 50/60-Hz line noise

<font color="#9bbb59">Standard: 1-32 Hz</font>
- Delta (1–4 Hz)
- theta (4–8 Hz) 
- Low beta (up to ~32 Hz) for possible higher-level attentional effects.
	
### **2. Downsampling**
**Goal**: reduce computational time during training + needs to match speech envelope. Often combined with anti-aliasing filter or a simple LP filter (below Nyquist frequency)		

<font color="#9bbb59">Standard:  to 125 Hz</font>

### **3. Normalizing and re-referencing**
**Z-score per channel:** All data within the same subject should be normalized together, not separately

**Re-reference to average or mastoids**
- To 1 channel (mastoid): enhance neural activity in a region of interest
- To mean of all channels: by subtracting the mean over all channels from each individual channel

--> <font color="#9bbb59">Since working with multiple datasets, all datasets re-referenced to average reference.</font>

### **(4. Artifact removal)**
ICA= independent component analysis
MWF= multichannel wiener filtering
	<font color="#d99694">Be careful as this can make info dissapear</font>
    
----

## <font color="#974806">Audio Preprocessing</font>
<span style="background:rgba(163, 67, 31, 0.2)">Audio envelope extraction: using compressed subband envelopes, resembling the processing of speech signals in the human auditory system. Makes it more interpretable and better aligned with envelope-following responses (EFRs) in EEG.</span>

### **1. Gammatone filterbank**
audio is split into many narrow “frequency channels,” each representing a small part of the speech spectrum.
simulates cochlear processing: like different hair cells in the inner ear

Equivalent rectangular bandwidth (ERB) =<font color="#9bbb59"> 1.5 Hz</font>
frequency band: <font color="#9bbb59">150 Hz-4 kHz</font>

### **2. Hilbert transform**
amplitude envelope extraction per subband

### **3. nonlinear compression**
This reflects loudness perception, humans perceive intensity nonlinearly 

e<sub>compressed</sub>​(t)=\[e<sub>raw</sub>​(t)]<sup>0.6</sup>

### **4. Filtering and resampling**
- **low-pass filter** at 50 Hz: EEG cannot track faster modulations
- **Downsample** to 125 Hz



    