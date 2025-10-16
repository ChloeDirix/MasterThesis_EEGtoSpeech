Key neural fact often leveraged: cortical (electrical) activity in auditory cortex (and higher auditory areas) tracks features of continuous speech (notably the amplitude envelope) and this tracking is modulated by attention.

Entrainment is general, but **speech comprehension modulates its strength**

**Timeline:**

| Year     | Study                    | title                                                                                            | Main Contribution                                                                          |
| -------- | ------------------------ | ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| 1953     | Cherry                   | /                                                                                                | the **“cocktail-party problem”**.                                                          |
| 2012     | Ding & Simon             | Emergence of neural encoding of auditory objects while listening to competing speakers           | there is cortical entrainment                                                              |
| **2012** | Ding & Simon             | Cortical entrainment to continuous speech: functional roles and interpretations                  | function of cortical entrainment: a few hypotheses                                         |
| **2014** | O’Sullivan et al.        | Attentional Selection in a Cocktail Party Environment Can Be Decoded from Single-Trial EEG       | first backward model                                                                       |
| **2014** | Ding, Chatterjee & Simon | Robust Cortical Entrainment to the Speech Envelope Relies on the Spectro-temporal Fine Structure | Cortical entrainment relies on **spectro-temporal fine structure**, not just the envelope. |


# O'Sullivan et al

**Cocktail-party problem**:
The human ability to focus auditory attention on a single speaker amid a mixture of background noise and conversations. It is an issue of selective attention

> 	Cognitive scientist Colin Cherry in 1953, highlights the difficulty technology has traditionally faced in replicating this seemingly effortless human skill
    

**how does the brain do it?** 
The brain **splits and organizes sounds into separate “streams”**, for example, one stream per speaker, then chooses one stream to enhance.

> **Auditory Scene Analysis (ASA)**: the study of how the auditory system transforms a mixture of sounds from multiple sources into **distinct perceptual representations**.  
> **Auditory objects**: perceptual units formed by detecting, extracting, and grouping spatial, spectral, and temporal regularities in the acoustic environment.  
> **Auditory stream**: a sequence of auditory objects over time.  
> **Auditory categorization**: the ability to reason about and respond adaptively to sounds in the environment.


**Features that help form auditory streams:**

| **Feature type**                                            | **Function**                                                                                   |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Binaural cues** (differences between left and right ears) | Localize sounds in space.                                                                      |
| **Spectral cues** (tone and timbre)                         | Separate sounds with different frequencies or voice qualities.                                 |
| **Temporal cues** (timing and rhythm)                       | Group sounds that change together over time.                                                   |
| **Pitch / fundamental frequency**                           | Track a single voice’s pitch pattern to distinguish it from others.                            |
| **Amplitude modulation**                                    | Follow changes in loudness; sounds with similar fluctuations likely come from the same source. |
| **Predictive regularities**                                 | Predict what comes next; unexpected changes can trigger attention shifts.                      |


# Ding & Simon 2
Function of cortical entrainment to speech? 
**Cortical entrainment** = neural marker showing how the brain tracks speech in real time.

| **Hypothesis**                     | **What it says the brain does**                                                                                  | **Processing stage**       |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------- | -------------------------- |
| **1. Onset tracking**              | Brain responds to sudden changes in sound intensity (edges/transients).                                          | Early auditory (bottom-up) |
| **2. Collective feature tracking** | Tracks multiple acoustic features (pitch, timbre, etc.) that fluctuate together; envelope represents the sum.    | Early–mid level            |
| **3. Syllabic parsing**            | Groups sounds into syllable-sized chunks using theta rhythms (~4–8 Hz), aligning one oscillation per syllable.   | Mid–high (linguistic)      |
| **4. Sensory selection**           | Uses rhythmic entrainment to select and enhance features of the attended speaker — “sampling” the right moments. | Attention/top-down         |

> These mechanisms likely **co-occur**, in different regions and stages — from raw auditory encoding to higher-level language comprehension.

---

**Frequencies of cortical entrainment**

| **Band**           | **Frequency (Hz)**       | **Linked function**                            |
| ------------------ | ------------------------ | ---------------------------------------------- |
| **Theta (4–8 Hz)** | Syllabic rhythm          | Speech understanding / intelligibility         |
| **Delta (1–4 Hz)** | Slower rhythms (phrases) | General rhythm, speech segmentation, structure |


**Is entrainment only for speech?**
No, but for speech, entrainment is modulated by attention and comprehension, making it highly useful for decoding selective attention.



# Ding, Chatterjee & Simon (2014)

**Core questions:**
Does robust cortical entrainment in noise reflect simple envelope tracking or an **object-level representation** relying on spectro-temporal fine structure?

**Methodology:**
- **Stimuli:** Natural speech vs. noise-vocoded speech (envelope preserved, fine structure degraded).
- **Measurement:** MEG/EEG tracking of cortical responses (delta/theta bands, TRFs).

**Main findings:**

**1. Spectro-temporal fine structure is critical**
- Natural speech: robust entrainment in noise.
- Vocoded speech: entrainment degraded, even with envelope intact.
Envelope tracking alone is not sufficient for noise-robust representations
    

**2. Delta-band entrainment predicts intelligibility:**
Reflects **object-level speech representation**, linked to comprehension
    

**3. Temporal response function (TRF) findings:**

|TRF component|Latency|Function|
|---|---|---|
|M50TRF|~50 ms|Early sensory processing; sensitive to noise & spectral degradation|
|M100TRF|~100 ms|Object-level processing; robust across noise & spectral degradation|


**Discussion**
- **Robust entrainment is object-based, not just stimulus-based**: natural speech is not affected by noise, indicating it is separated from noise and stored as an individual auditory object.
- Contrast gain control contributes but cannot explain the robustness alone.

- The brain does not do simple bottom-up envelope tracking but **analysis-by-synthesis approach**.  Degradation of fine structure disrupts this grouping, weakening cortical entrainment.
		- *analysis*: the sensory system breaks up the sensory input into fundamental features (pitch, formants, binaural cues)
		- *Synthesis*: Features belonging to the same speech stream are bound into an auditory object.


