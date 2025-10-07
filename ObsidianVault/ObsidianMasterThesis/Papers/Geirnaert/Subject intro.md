## Decoding selective auditory attention from EEG to speech using contrastive learning with transformers

## Inhoud

**Context**

People with hearing impairment often have difficulties understanding speech in noisy environments, leading to decreased quality of life. This can be overcome by assistive hearing devices such as hearing aids and cochlear implants, which contain noise reduction algorithms that extract the sound from a target direction while reducing interfering sound.  
  
However, in a so-called `cocktail party' scenario where multiple sound sources are present, a fundamental problem appears: how does the assistive hearing device know which speech signal the listener is actually attending to? We refer to this problem as selective auditory attention decoding (AAD).  
  
By equipping a hearing device with EEG sensors, it is possible to extract attention-related information directly from where it originates, i.e., in the brain. While various methods have been developed for EEG-based AAD, they either do not reach high enough accuracy when using few time samples (i.e., a short decision window length, required for detecting attention switches), or do not generalize well to new data recorded in different conditions [1].

[1] Geirnaert, Simon, et al. "Electroencephalography-based auditory attention decoding: Toward neurosteered hearing devices." IEEE Signal Processing Magazine 38.4 (2021): 89-102.

**Objectives**

In this thesis, we want to develop new deep learning-based AAD algorithms that achieve a high accuracy at short decision windows and generalize well over time and across participants. As a framework, we will adopt contrastive learning and transformers, which have been successfully used in single-speaker data [2], but not (yet) in decoding selective attention to multiple speakers. We will start from the algorithm proposed in Bollens et al. [2] and use transfer learning techniques to finetune the model on selective attention datasets. Special care will be taken towards the evaluation of the developed algorithms, which is crucial in determining the added value of deep learning-based AAD algorithms [3].  
  
For this thesis, the student should be proficient in signal processing and machine learning. You will be guided by an expert in AAD algorithms and an expert in deep learning.  
  
[2] Bollens, Lies, Bernd Accou, and Tom Francart. "Contrastive Representation Learning withTransformers for Robust Auditory EEG Decoding." (2024).  
  
[3] Puffay, Corentin, et al. "Relating EEG to continuous speech using deep neural networks: a review." Journal of Neural Engineering 20.4 (2023): 041003.
![[ProblemStatement.png]]


https://icts.kuleuven.be/masterproeven/student/publication/thesissubjects/94705
