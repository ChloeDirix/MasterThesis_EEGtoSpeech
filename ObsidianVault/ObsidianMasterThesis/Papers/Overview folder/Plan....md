## 1. Data

- **Dataset**: combine datasets      
- **Subject-specific vs subject-independent (cross-subject) decoding?**
		contrastive learning benefits from subj-indep (chatgpt) + it's needed 
        
- **Stimuli**
    - Binary left/right attention vs multi-locus attention
    - Use of multiple stories/speakers (leave-one-story+speaker-out strategy)
        

---

## 2️ Preprocessing Choices

1. **EEG preprocessing**??

    - Band-pass filtering 
		    **standard: 1–32** or devide in beta, ...
    - Downsampling rate
    - Normalization or **z-scoring** per channel
        
2. **Segmentation**

    - Length of EEG windows (0.1 s → 10 s)
		    T+C: allow short: 0.5-2s
    - Overlap between windows (**sliding** vs non-overlapping).
		    sliding preferred for real-time evaluation
        
3. **Audio preprocessing**
    
    - Envelope extraction or learned embeddings (e.g., Mel spectrogram).
    

---

## 3️ Model Architecture Choices

1. **EEG encoder**
    
    - Transformer-based (temporal attention, multi-head, number of layers/heads).
		    
    - Optional CNN/Conv1D front-end for feature extraction
			Start with temporal transformer; optional to experiment with multi-head spatial attention over electrodes.
        
2. **Audio encoder**
    
    - CNN or transformer to extract features from speech envelopes.
    - Dimensionality of embeddings.
	        Simple CNN or linear projection is enough; focus on embeddings aligned with EEG for contrastive loss.
1. **Fusion strategy**
    
    - Concatenation vs cross-attention between EEG & audio embeddings.
	        Concatenation is simplest; cross-attention may improve performance and interpretability.
	        
1. **Output**
    
    - Binary classification (left/right) or similarity score to candidate speech.
    - Embedding space for contrastive learning.
	        Contrastive pretraining → classification or nearest-neighbor selection. Embeddings are central to your contribution.
	    
**Advice:** Focus first on **EEG + audio → transformer → embedding space** pipeline. Later you can add cross-attention, multi-head electrode attention, or hybrid CNN layers.

---

## 4️ Training Choices

1. **Loss function**
    
    - Contrastive loss (InfoNCE, NT-Xent) vs supervised cross-entropy.
    - Hybrid: pretrain contrastive → fine-tune classifier.
	        Contrastive loss is central. You can optionally combine with supervised cross-entropy for final classification.
	        
2. **Batching strategy**
    
    - Positive/negative pair selection for contrastive learning.
    - Window length and number of examples per batch.
	        Positive/negative pairs must be carefully defined for contrastive learning.
	        
1. ***Regularization***
    
    - Dropout, weight decay, data augmentation (add noise, other subjects).
	        Dropout, weight decay useful if overfitting; start simple.

	        
2. ***Pretraining***
    
    - Contrastive pretraining vs training from scratch.
	        Pretraining with contrastive loss first is recommended, but you can test end-to-end supervised as a baseline.

---

## 5️ Evaluation Choices

Evaluation metrics?
- Classification accuracy (attended speaker)
- Correlation of EEG–speech embeddings
- Generalization (cross-subject)
- Data efficiency (performance vs training set size)
		
1. **Metrics**
    
    - Decoding accuracy (% windows correct)
    - Correlation between EEG embedding & attended speech embedding
    - Cross-subject generalization
    - Latency / decision window performance
	        Standard metrics for benchmarking and supervisor-friendly comparisons.
1. **Validation**
    
    - Leave-one-story+speaker-out
    - Cross-validation for subject-specific models
		    Avoid overfitting to specific stimuli.
        
1. ***Ablation studies***
    
    - Which EEG channels, frequency bands, attention heads are most informative.
    - Performance vs decision window length.
		    Helps interpretability; do if time allows
        

---

## 6️ *Optional Extensions / Innovations*

1. ***Hybrid architectures***
    
    - *CNN + transformer pipeline for improved feature extraction.*
	        Only if transformer alone underperforms or short-window performance needs boost.
	        
1. ***Spatial attention***
    
    - *Multi-head attention over electrodes for dynamic channel weighting.*
	        Useful for interpretability and potentially better cross-subject generalization.
	        
2. ***Frequency-band decomposition***
    
    - *Separate EEG bands as streams for attention.*
	        Adds insight into EEG frequency importance; good for ablations.
	        
3. ***Real-time decoding***
    
    - *Sliding-window streaming setup, online prediction.*
	        Very nice for applications, but not required initially.
	        
4. ***Cross-subject / domain adaptation***
    
    - *Adversarial or contrastive methods to create subject-invariant embeddings.*
	        

5. ***Multi-locus or multi-modal fusion***
    
    - *Beyond left/right; combine EEG with eye-tracking or head orientation.*
	        Only pursue if dataset allows
6. ***Interpretability***
    
    - *Visualize attention weights, embeddings, and contribution of channels/bands.*
			Valuable for discussion/figures but not required to demonstrate novelty.
