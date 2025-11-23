# <font color="#4bacc6">Plan </font>

---
## <font color="#0070c0">Data</font>

### 1. **Dataset**
Goal: Generalising across completely different datasets 
--> comes with challenges: see [[Preprocessing]]

### 2 **Subject-specific vs subject-independent (cross-subject) decoding?**
([[Subject dependent or specific vs subject independent]])

- SS training show the model’s capacity to learn individualized patterns <font color="#c0504d">+ finetuning</font>
- SI training proves generalisation

 => Both needed

### 3. **Avoid data leakage**
- leave-one-story out: test on held out story
- leave one speaker-out: test on unseen speaker 
- leave one session out: test on unseen session
- leave one subject out: test on unseen subject 
- leave one-dataset out: test on unseen recording setup


---

## <font color="#4f81bd">Baselines</font>
- LSR
- NSR
- end-to-end algorithm: could be algorithm without contrastive learning

---

## <font color="#0070c0">Preprocessing Choices</font>
### 1. **EEG preprocessing & Audio preprocessing**
[[Preprocessing]]
       
### 2. **Segmentation**
**length of windows**: Length of EEG windows (0.5 s → 20 s) 
- <font color="#4bacc6">longer windows</font>: help contrastive representation learning
- <font color="#4bacc6">shorter windows</font> are better for real-time inference
--> Test multiple lengths 

**overlapping vs non-overlapping**: sliding preferred for real-time evaluation. Non-overlapping probably enough?



---

## <font color="#4f81bd">Model Architecture Choices</font>
[[Notes Bollens et al. --- Contrastive Representation Learning with Transformers for Robust Auditory EEG Decoding]]
![[ModelArchitecture.png]]

### 1. **EEG encoder**
(Details: [[EncoderSteps]])

_Why Transformer?_ Superior capture of temporal relations and flexible receptive field. Bollens et al. showed transformer + contrastive helps EEG robustness.

### 2. **Audio encoder**  
-- focussed on making it compatible with EEG
-- Conv + LSTM
	        
### 3. **Contrastive loss**
(details: [[Contrastive Loss Steps]])

Positive pair: (EEG, _attended audio_)
Negative pair: (EEG, _unattended audio_)
	    
---

## <font color="#4f81bd">Training Choices</font>

### **Batching strategy**
- **Batch composition:** each batch contains many **(EEG window, attended audio, unattended audio)** triplets across multiple subjects and across datasets
- **In-batch negatives**: include cross-subject, cross-trial and cross-dataset negatives
- **Hard negatives:** same speaker but unattended (be careful as it can destabilize!)
- **Balanced dataset**s --> avoid bigger datasets dominating and ensure each dataset appears in train/val/test
	        
---

## <font color="#4f81bd">Evaluation Choices</font>

### **Evaluation metrics?**
- **Accuracy (% windows correct)** for attended speaker identification
- **Report per dataset performance**: run model on validation/test set of every dataset
- **MESD (Minimal Expected Switching Duration)**

### **Validation**
- 8-fold cross-validation for subject-specific models
- LOSO for Subject-independent

--> + other cross validations (see *data*)
        
### ***Ablation studies: if time allows***
- Which EEG channels, frequency bands, attention heads that are most informative, without adapter vs with adapter, single dataset vs joint datasets
        

---

## <font color="#4f81bd">6️ Extensions: maybe future research</font>

1. ***Spatial attention***
    - *Multi-head attention over electrodes for dynamic channel weighting.*
		        
2. ***Real-time decoding***
    - *Sliding-window instead of non-overlapping* 

3. ***multi-modal fusion***
    - *combine EEG with eye-tracking or head orientation*
	    
4. ***Interpretability***
    - *Visualize attention weights, embeddings, and contribution of channels/bands.*
		
