
---
## **Cocktail party problem solution?**
see [[Notes Intro]]

- **Advanced system processing algorithms**
		suppress background noise, but still major difficulties in cocktail party scenario

- **Beamforming algorithms**
		Use microphone array signals to suppress background noise and extract a single speaker

	*But*: *which speaker is the attended speaker?*
		- <span style="font-weight:bold; color:rgb(75, 145, 127)">Simple heuristics </span>--> selecting the loudest speaker or assuming the attended speaker is in front of the listener. ==Problems:== car ride, listening to public address system,...
		- <span style="font-weight:bold; color:rgb(75, 145, 127)">neuro steered hearing devices</span>

---

## **Neuro Steered hearing devices**
The brain synchronizes with the speech envelope --> we can extract attention related info from the brain directly

<span style="font-weight:bold; color:rgb(236, 81, 112)">= ‘auditory attention decoding’ (AAD) problem</span>

##### Which neurorecording technique?
- ECOG: invasive
- MEG: high cost and lack of wearability 
- EEG: a non-invasive, wearable, and relatively cheap 

##### Main idea of the AAD algorithm
Determine who the attended speaker is, starting from a multichannel EEG, in a multi-speaker (sometimes noisy environment).


![[neuro-steered hearing device.png]]


AAD-block = algorithm that determines the attended speaker by integrating the demixed speech envelopes and the EEG (comparison between speech envelope and speaker)

*But what AAD algorithms are the most efficient?*

---

## **AAD algorithms**

##### Remarks
- only methods before 2020
- two speakers
- make abstraction of the speaker separation and denoising block --> assume that the AAD block has direct access to the envelopes of the original unmixed speech source
- Types: [[Types of AAD algorithms]]

##### The different algorithms
	
![[Different AAD algorithms.png]]


#### <span style="background:rgba(240, 200, 0, 0.2)">Linear methods</span>
In this study: all use backward modeling:

The decoder d<sub>c</sub>(l) is a linear time invariant spatio-temporal filter on the EEG
- x<sub>c</sub>(t) =  C-channel EEG data
- s<sub>a</sub>(t) = Speech envelope
![[linear AADs.png]]


- c= channel index (1:C)
- l= time lag index (0:L-1)
- L=per channel filter length


<u>Different algorithms</u>
They all have a different way to find d
1) Supervised minimum mean-squared error backward modelling ([[MMSE]])
2) Canonical correlation analysis ([[CCA]])
3) Training-free MMSE-based with lasso ([[MMSE-adap-lasso]])

<u>Multiple speakers? </u>
by correlating the reconstructed speech envelope with all additional speech envelopes of the individual competing speakers and taking the maximum

#### <span style="background:rgba(240, 200, 0, 0.2)">Nonlinear methods</span>
In this study: based on deep neural networks 
	<font color="#9bbb59">+</font> Approach can be similar to linear or can use direct classification
	<font color="#c0504d">-</font> vulnerable to overfitting (small datasets)

Types?
- Fully connected stimulus reconstruction neural network ([[NN-SR]])
- Convolutional neural network to compute similarity between EEG and stimulus ([[CNN-sim]])
- Convolutional neural network to determine spatial locus of attention ([[CNN-loc]])

---

## **Comparative study**
compared all algorithms on two available datasets
- in subject specific way
- all algorithms were retrained from scratch for each dataset
- performance evaluated via **accuracy** p

<u>But</u>
EEG = many different neural processes
--- correlation between reconstructed and attended envelope is quite low
--- decision window  τ must be long. (<span style="background:rgba(136, 49, 204, 0.2)">p ~ τ</span>)  

*<font color="#7030a0">p(τ )-performance curve</font> = tradeoff between accuracy and decision delay


<u>MESD</u> = *minimal expected switch duration*
determines the most optimal point on the p(τ )-performance curve in the context of attention-steered gain control by minimizing the expected time it takes to switch the gain between two speakers in an optimized robust gain control system

<font color="#76923c">+</font> single number time metric: 
	<font color="#76923c">+</font> no statistical power loss due to multiple comparison corrections
	<font color="#76923c">+</font> comparison is directly based on most relevant points of the performance curve

**Remarks**: 
- it's a comparative metric, does not reflect the true switching time.
- higher MESD = worse AAD performance

---
## **Statistical analysis - results**

linear mixed-effects model (LMM) on the AAD algorithms used for comparison

##### **Performance curves**

![[performance.png]]

| --                              | --                                                                                                                                            |
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| CCA                             | consistently outperformed other linear methods across datasets                                                                                |
| CNN-loc                         | strong performance on Das-2015 (at short decision windows (<10 s)), but performed poorly on Fuglsang-2018 and showed high subject variability |
| MMSE-adap-lasso, CNN-sim, NN-SR | did not exceed the significance level and were excluded                                                                                       |
|                                 |                                                                                                                                               |

<span style="background:rgba(240, 107, 5, 0.2)">Regularization type</span>: (ridge vs. lasso) had no significant impact.
<span style="background:rgba(240, 107, 5, 0.2)">Averaging correlation matrices</span> (early integration) outperformed averaging decoders (late integration)

##### **Subject-specific MESD**
![[Subjectperformance.png]]

Same conclusions as performance curves


##### **Discussion:**
**<span style="background:rgba(3, 135, 102, 0.2)">CCA</span>** remains the most reliable method, outperforming traditional linear decoders.
<span style="background:rgba(3, 135, 102, 0.2)">CNN-loc </span>: potential for short windows but lacks robustness across datasets => **generalization issues**. probably due to noise-susceptibility

Nonlinear neural network methods in general failed to generalize well,  need for<font color="#92cddc"> larger, more diverse datasets</font> and<font color="#92cddc"> cross-validation</font> to avoid overfitting. there are also blackbox methods and it remains unclear what causes succes or failure on a certain database. The current datasets are probably also to small to draw decent conclusions. But still there is a big risk of overfitting

--- 

## **Open Challenges**

####  A. Validation in realistic listening scenarios instead of controlled lab conditions

- more than two speakers: **unclear**
- more noise and reverberation: **moderate noise gives better accuracy then no noise**
- attention switches: **unclear**

#### B. Effects of speaker separation and denoising algorithms

<u>speaker separation?</u>
Most AAD algorithms need clean speech envelopes from individual speakers 
=> <font color="#92cddc">speaker separation</font> from microphone recordings possible, but imperfect separation degrades performance. Algorithms that do **not rely on speech envelopes** (e.g., CNN-loc) have an advantage here.

<u>effect on performance?</u>
minor AAD performance loss when combining separation/denoising with AAD

<font color="#c0504d">Joint optimization</font> approaches: of denoising and speaker separation are promising

#### C. EEG miniaturization and wearability

tradeoff between too bulky/wet and performance (nr of channels)

size?
- around the ear approach: lower performance
- Strategic electrode placement: can preserve accuracy with fewer channels

wet?
Dry EEG systems are more practical and can have good performance. But more research needed

#### D. Outlook

EEG-based auditory attention decoding (AAD) is possible using CCA for example. But accuracy at short decision windows is too low! Not possible in real time scenario

Decoding spatial attention: faster responses but lacks robustness across datasets

---
# <font color="#c0504d">Future challenges:</font>

- Algorithms that improve short-window accuracy and generalise well
- Develop unsupervised or adaptive methods to eliminate per-user training and adapt to changing EEG signals.
- Build closed-loop neuro-steered hearing devices, where user feedback interacts with the algorithm to improve performance.
- Test algorithms in real-world scenarios with real users.