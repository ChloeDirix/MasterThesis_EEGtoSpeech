# Vandecapelle et al. (2021)

##### **CNN?**
- don't require clean speech envelopes
- can be used to locate locus of auditory attention (interpretable features)
- does not rely on linear time invariant assumption

it's showed that a (subject-dependent) CNN using a classification approach can outperform linear methods for decision windows of 10 s, but the performance drastically drops when using shorter windows

##### **Architecture**
![[image.png]]
- Input: matrix of size **64 × T** (64 channels, T time samples)
- First convolutional layer: _spatio-temporal filters_ of size 64 × 17 (i.e. across all channels, 17-time-sample window) → yields 5 time series (i.e. 5 filters) over T
- Activation: ReLU
- Then average pooling along time → each time series is reduced to one scalar
- Two fully connected (FC) layers follow: first with 5 neurons (one per filter output), then final layer with 2 outputs (left/right)
- Loss: cross-entropy for classification
- The total number of trainable parameters is ~5,500


**interesting points:**

***subject-specific decoders**,* but data from _other subjects_ are included in training (excluding the held-out test segments) to act as a regularizer / data augmentation. 

***trade-off*:** accuracy and decision window length (Geirnaert et al., 2020) 

***spatial locus of attention*** 
- (lu et al.): using entropy features: insufficient
- recent research: direction of auditory attention is neurally encoded

Architecture: ***spatio-temporal kernels across all channels + a temporal window*** : each convolution filter is 64 (spatial) × 17 (time) in their implementation. 
    
Architecture: ***Reduction of temporal dimension via average pooling*** and then do classification. This architecture is simple and interpretable, yet effective.

***leave-one-story+speaker-out***: test data have unseen stories and speakers) to avoid overfitting to stimulus identity.

***ablation-like tests*:** test which EEG frequency bands are important

***weight mapping across channels***: look at spatial importance by examining convolutional filter weights per channel (averaged). These weight maps give hints about which EEG channels are informative

**performance:**
- outperforms a baseline linear model under many window lengths
- comparisons to linear methods are not entirely “fair,” because the CNN is free to extract arbitrary features (not constrained to link to speech envelopes) and also different preprocessing is used
- inter-subject variability or dependence on stimuli.





##### **key insights**
- **Subject-specific decoders** benefit from including data from other subjects as regularization / augmentation.
- **Accuracy vs decision window trade-off**s are important for benchmarking.
- **Leave-one-story + speaker-out cross-validation** is critical to avoid overfitting to specific stimuli.
- **Frequency bands** (delta, theta, alpha, beta) matter; ablations can reveal which bands are important.



**Ciccarelli et al. 2019**

compare O'sullivan's TRF 
- DNN to make a predicted audio
- CNN to to direct classification
    

##### linear model: 
![[image-1.png]]

##### DNN
![[image-2.png]]
    
    
##### CNN
-- direct classification


##### useful parts
- benchmark comparison: results give you a baseline
- Their **end-to-end CNN classifier** design (EEG + audio input → attention label) is compact and interpretable. It’s a solid template if you want to design new AAD networks.
- detailed preprocessing
