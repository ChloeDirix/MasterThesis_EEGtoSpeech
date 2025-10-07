[[Subject intro]]

**Problem**: difficulties in noisy environment with hearing aids: <font color="#92d050">'cocktail party' scenario'</font>

**Solution 1**a: hearing aids with noise reduction
	***But***: Still major difficulties as person cannot focus on 1 specific conversation. 

**Solution 1**b: Beamforming Algorithms 
	reception is steered towards a specific direction
	**But** how to know which signal the person wants to hear in 
		<font color="#31859b">=AAD=auditory attention decoding problem</font>

**Solution 2**a: simple heuristics
	--> selecting the loudest speaker or assuming the attended speaker is in front of the listener. 
	***But***: car ride, listening to public address system,...
	
**Solution 2**b: Neuro-Steered hearing devices: 
	--> equip hearing device with EEG sensors
	***But***: 
		What algorithm to use to predict attended speaker from EEG

**Solution 3**a: Many algorithms (linear, nonlinear, DL)
	***But***
	- low accuracy when using few time samples
	- don't generalize well

**Solution 3**: Contrastive learning and transformers

**method**
- Bollens et al. --> succesfully used for single speaker data
- use transfer techniques to finetune model for multispeaker
- evaluation

**Objectives**: 
- higher accuracy
- better generalisation