| Aspect                | **Subject-Specific (SS)**                                                                                                                                                   | **Subject-Independent (SI)**                                                                  |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Definition**        | A model trained and tested on data from the **same person**.                                                                                                                | A model trained on data from **many people** and tested on a **new, unseen person**.          |
| **Goal**              | Optimize decoding for one individual<br><font color="#8064a2">personalized brain model</font>                                                                               | Generalize decoding rules across people<br><font color="#8064a2">universal brain model</font> |
| **Training Data**     | Uses that person’s EEG trials only                                                                                                                                          | Uses EEG data pooled from all other subjects                                                  |
| **Evaluation Scheme** | Cross-validation within one subject                                                                                                                                         | Leave-one-subject-out (LOSO) — each subject is left out once for testing                      |
| **Disadvantage**      | -- Data from one single person only --> more overfitting<br><br>-- less desirable in real world application - need per user calibration (implies hours of calibration data) | lower accuracy as brain signals always vary across people                                     |

Contrastive learning: improves SI generalisation

**Expected in research**
- Show both 
	-- SS → show the model’s capacity to learn individualized patterns 
	-- SI → show its potential for generalization and real-world usability
- Prioritize interpretability and neurophysiological plausibility for scientific validity


**SS: fine-tuning needed:** 
1.  **train the Subject-Independent (SI)** model (which uses data from all subjects)
2. **start from that pretrained SI model** and **fine-tune** it on the small dataset for the specific subject (SS model).
3. Use a **smaller learning rate** (gentler adjustments) during fine-tuning


**Ultimate goal**
fully **subject-independent**, **real-time AAD system** that:
- Works instantly for new users
- Adapts dynamically to changing attention
- Can control hearing aid focus or auditory scene separation seamlessly