# Crosse et al. (2016) — _The mTRF Toolbox_
“Understanding how brains process sensory signals in natural environments is one of the key goals of twenty-first century neuroscience” (Crosse et al., 2016, p. 1)

> _How does the brain process sensory signals in natural environments?_

###### **timeline**
1. **Event-Related Potentials (ERPs)**
	- Traditional approach: averaging EEG responses to many repetitions of brief, discrete, time-locked stimuli.
    - Objective: estimate the system’s impulse response function.
    - Limitation: Works only for short, isolated events and does not capture ongoing neural processing of natural, dynamic stimuli (like real speech). 

2. **Continuous Modeling -- system identification (SI)**: 
		- Sees brain as system that transforms sensory inputs --> neural responses.
		- By estimating how the system responds to a continuous input, one can infer its impulse response (**temporal response function TRF**).
		--> linear time invariant systems		

| type                                     | description<br>                                                                                                                                |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| early version: reverse correlation       | cross-corelate input and output signals to estimate an impulse response<br>-- works for random/white noise<br>-- struggles with natural speech |
| later: regularized regression approaches | ridge regression, boosting, normalized reverse correlation), allowing unbiased estimation of TRFs                                              |

| Type           | Description                                  | Purpose                                                                                                                                                                                                                                                                                                                              |
| -------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Forward model  | Predicts EEG response from stimulus features | Good for interpreting neural tuning (physiological insight)                                                                                                                                                                                                                                                                          |
| Backward model | Reconstructs stimulus features from EEG      | Good for decoding / attention tracking<br>Backward models (used in **stimulus reconstruction**) are especially powerful for multichannel data like EEG:<br>++ They leverage correlations across channels.<br>++ They’re robust to noise and redundancy. <br>++ They allow decoding of what stimulus features were being attended to. |


3. **mTRF toolbox:** 
	- open-source, standardized implementation of SI using regularized linear (ridge) regression.
    - Enables multivariate modeling (mTRF), handling all EEG channels simultaneously.
    - Facilitates both **forward** (encoding) and **backward** (decoding) TRF estimation.

	 Key benefits:
        - Accessible and reproducible EEG–speech modeling.
        - Optimized regularization for different datasets.
        - Demonstrates practical use cases for auditory and visual processing. 

	
