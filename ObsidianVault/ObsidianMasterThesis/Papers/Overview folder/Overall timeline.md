
**1953**: Colin cherry: *"the cocktail-party problem"*
**2012**: Ding & Simon: there is cortical entrainment


### The Classic “Linear Model” Approach (2012)

**2012**: Mesgarani & Chang: backward models
**2014**: O'Sullivan's backward model

---

### Deep Neural Networks (DNNs) (2017)

*Why?*  
Because EEG signals are noisy and the mapping to speech features is complex — nonlinear models could, in theory, capture richer dependencies.

#### Recurring families
###### feedforward networks (MLPs) (2017)
idea: estimate speech envelope with a multilayer perceptron

> **2017**: de Taillez et al.: Machine learning for decoding listeners’ attention from electroencephalography evoked by continuous speech

✅ Easy, interpretable.  
❌ not much better than linear

---
###### CNNs (2019)

idea: EEG has _spatial_ and _temporal_ patterns — convolutional filters can capture these.

> **2019**: Ciccarelli et al.: comparison of two-talker Attention Decoding from EEG with nonlinear neural networks and  Linear Methods
> **2021**: Vandecapelle: EEG-based detection of the locus of auditory attention with convolutional neural networks

2 types:
- **Stimulus reconstruction CNNs**: not popular
- **Classification CNNs   
    = _end-to-end model_(no stimulus reconstruction)

✅ Much better accuracy
❌ Needs lots of data, less interpretable.

--- 

###### Recurrent neural networks (RNNs) and LSTMs (2021)
idea: EEG and speech are **temporal sequences** --> models that can capture dependencies over time.

> 2022: Xu et al.: Auditory attention decoding from EEG-based Mandarin speech envelope reconstruction    


###### Hybrid models: CNN + RNN
Idea:
- CNN feature extraction
- RNN: temporal integration

> **2021**: Kuruvila et al. Extracting the Auditory Attention in a Dual-Speaker Scenario From EEG Using a Joint CNN-LSTM Model


✅ Better for sequential structure  
❌ poor generalisation

---

###### Attention mechanisms/ transformers (2022-2024)
idea: model long range dependencies across time and across EEG channels
Transformers can capture **non-local EEG dependencies**, which may correspond to distributed neural processes (frontal, auditory, parietal).
Contrastive learning is _self-supervised_ — it doesn’t need explicit labels like “attend left/right.”  
Instead, it **learns a shared representation** between EEG and audio features by bringing related pairs closer and pushing unrelated ones apart.


###### AADNet (2025)
Direct classification, improved feature extraction, can incorporate contrastive/self-supervised learning

> 2025: _“AADNet: An End-to-End Deep Learning Model for Auditory Attention Decoding”_




---
analogy💡 

- Linear → bicycle (baseline)
- CNN → early car (end-to-end)
- RNN → car with cruise control (temporal modeling)
- Hybrid CNN-RNN → sports car (combined features)
- Transformers → AI-assisted self-driving (long-range dependencies)
- AADNet → modern electric car with optional AI features (refined architecture, can include self-driving / transformers / contrastive learning)