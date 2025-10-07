Problem: [[Notes Intro]]

**papers**
Start with the neuroscience + baseline methods, then move to contrastive & transformers.

## **1️⃣ Neuroscience / Foundational Theory**

These papers explain _why auditory attention decoding is possible_ and the neural mechanisms behind it.

| Paper                                                                             | Focus                                         | Relevance / Why Important                                                     | Priority |
| --------------------------------------------------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------- | -------- |
| O’Sullivan et al., 2015 – _Attentional selection in a cocktail party environment_ | Backward mTRF baseline, single-trial decoding | Seminal proof that EEG carries attention info; justifies reproducing baseline | ★★★      |
| Ding & Simon, 2014 – _Cortical entrainment to continuous speech_                  | Neural entrainment theory                     | Explains why EEG tracks attended speech; theoretical foundation               | ★★       |
| Vanthornhout et al., 2018 / Lalor & Foxe 2010 (optional)                          | Neural tracking of speech                     | Additional neuroscience grounding                                             | ★        |

**Takeaway:** Provides the “why” of AAD and baseline envelope reconstruction.

---

## **2️⃣ Baseline Methods / Classical AAD**

Linear or simple nonlinear baselines you need to reproduce before your own experiments.
## Summary — “If I just want a solid, reproducible preprocessing pipeline”

If you follow these three papers’ methods:

- O’Sullivan et al. (2015) (for filtering/referencing/downsampling)
    
- Biesmans et al. (2017) (for envelope extraction)
    
- Crosse et al. (2016) (for modeling setup)


citations for your “standard preprocessing”:

> EEG preprocessing followed the approach of O’Sullivan et al. (2015) and Biesmans et al. (2017). EEG data were band-pass filtered between 0.5–8 Hz, re-referenced to the average of the mastoids, and downsampled to 64 Hz. Speech envelopes were extracted using a gammatone filterbank, Hilbert envelopes, and power-law compression, as described in Biesmans et al. (2017).  
> Modeling was performed using the mTRF framework (Crosse et al., 2016).

|Paper|Focus|Relevance|Priority|
|---|---|---|---|
|Crosse et al., 2016 – _mTRF toolbox_|Linear backward/forward modeling|Canonical baseline, reproducible|★★★|
|Geirnaert et al., 2021 – _AAD review_|Overview of linear/nonlinear methods|Contextualizes your contribution|★★|
|Vandecappelle / Das et al., CNN baseline papers|CNN / RNN baselines for AAD|First nonlinear comparison|★★|
|Fast EEG-Based Decoding Using CSP|Lightweight directional decoding|Good for speed / baseline comparisons|★|
|Time-adaptive Unsupervised AAD|Unsupervised baseline|Inspiration for label-efficient methods|★★|

**Takeaway:** Gives benchmarks to compare against; essential for validating your models.

---

## **3️⃣ Data / Preprocessing / Features**

Covers datasets, preprocessing, and auditory-inspired features.

|Paper|Focus|Relevance|Priority|
|---|---|---|---|
|Accou – SPARRKULee|Dataset description & preprocessing tips|Provides EEG dataset & practical preprocessing advice|★★★|
|Biesmans et al., 2017|Auditory-inspired feature extraction|Guides envelope/spectrogram/gammatone representations|★★|
|Puffay et al.|Continuous speech decoding|Motivates use of naturalistic stimuli|★★|
|Ear-EEG / Audiovisual AAD papers|Real-world EEG acquisition|Motivation for practical/robust applications|★|

**Takeaway:** Helps you design preprocessing pipeline and choose features.

---

## **4️⃣ Contrastive / Self-Supervised Learning**

These are the _core modern methods_ for your approach.

|Paper|Focus|Relevance|Priority|
|---|---|---|---|
|“A Contrastive-Learning Approach for AAD”|Contrastive EEG-speech alignment|Directly aligned with your research|★★★|
|Auditory Attention Decoding with Task-Related Multi-View Contrastive Learning|Multi-view VAE + contrastive learning|Shows advanced EEG-speech contrastive setup|★★★|
|Self-Supervised Speech Representation & Contextual Embedding|Match-mismatch classification|Integrates EEG + speech + contrastive learning|★★★|
|SimCLR – Chen et al., 2020|Contrastive learning principles|Foundation for InfoNCE, projection head, augmentations|★★|
|Improving AAD in noisy environments with contrastive learning|Robustness under noise|Realistic scenario; informs ablations|★★|

**Takeaway:** Provides methodological blueprint for your contrastive learning pipeline.

---

## **5️⃣ Transformer / Deep Architectures**

Papers on transformers and modern deep models for EEG.

|Paper|Focus|Relevance|Priority|
|---|---|---|---|
|EEG-Transformer|Transformer for EEG (imagined speech)|Shows transformer application to EEG|★★★|
|AADNet|CNN-based spatiotemporal AAD|Useful deep learning baseline|★★|
|Attention Is All You Need|Transformer foundation|Fundamental architecture knowledge|★★★|

**Takeaway:** Provides the architectural knowledge needed for your transformer-based EEG decoder.

---

## **6️⃣ Robustness / Real-World / Unsupervised**

Focuses on real-world constraints, adaptive methods, and minimal labeling.

|Paper|Focus|Relevance|Priority|
|---|---|---|---|
|Unbiased Unsupervised Stimulus Reconstruction|Label-free decoding|Method inspiration for self-supervised setups|★★★|
|Time-adaptive Unsupervised AAD|Adaptive unsupervised decoder|Shows time-adaptive potential|★★|
|EEG-based AAD with audiovisual speech|Hearing-impaired, noisy env|Motivation for multimodal & robust AAD|★★|
|Ear-EEG in multi-speaker|Sparse electrodes|Realistic wearable AAD application|★|



