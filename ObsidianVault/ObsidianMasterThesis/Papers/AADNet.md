= end-to-end deep learning model (neural network) that utilizes the modified inception block 

as opposed to envelope-based AAD algorithms

#### **Baseline**
*linear*: most robust methods: LSR (backward), CCA
*non linear:*
> methods address different tasks 
> 	- single-speaker matchmismatch
> 	- AAD speaker identification (SpkI)
> 	- locus of attention (LoA))
> utilize different recording modalities 
> 	- EEG
> 	- MEG
> employ various speech features 
> 	- linguistics
> 	- speech envelope
> 	- spectrogram)
> 	
**Not comparable** --> careful selection of baseline!!!


#### **AADNet**

**background**: A neural network learns from data using **convolutional filters**, small windows that scan across your data to detect features (like shapes in images or patterns in sound).
##### Inception backbone:
![[image 1.png]]
= basic convolutional block
= smart building block for a neural network (invented by Google to make image recognition faster and better)

##### The Problem:
Normally, you have to decide what **size** of window (filter) to use:
- Small windows (like 1×1 or 3×3) catch fine details.
- Large windows (like 5×5) catch bigger patterns.
You don’t always know which is best! Why not use them all??

##### Inception Idea
**multiple parallel paths (branches)**:
- One path uses small filters (1×1)
		<font color="#205867">The transform branch</font>: transforms features from the earlier layers to the later layers
- One uses medium filters (3×3): <font color="#205867">Feature branch</font>:
- One uses large filters (5×5): 
		<font color="#205867">Feature branch</font>:  extract spatial features of the input of the current layer
- <font color="#205867">Pooling branch</font> = a way to summarize nearby values
	
Each branch looks at the same input in a slightly different way, and then the results are concatenated. This lets the network learn **features at multiple scales** without becoming too big or slow.

**Additional 1×1 convolutions?**
that reduce how many input channels (data dimensions) each branch handles, so model's computing complexity stays down

**ReLU**
output of the four branches passed through ReLU activation function 	


##### Architecture
![[image-1 1.png]]

Each branch has:
- A **Batch Normalization (BN)** layer → helps stabilize training
- A **Modified Inception Block** → extracts meaningful features
- A **Max Pooling layer (3×3)** → reduces the data size
- Another **BN layer**

**channel-wise pearson correlation** ==> <font color="#00b050">feature vector</font>
- **Dropout (DO)** → prevents overfitting
- **Fully Connected (FC) layer** → combines information
- **Softmax** → produces probabilities for each possible speaker 


##### why modify inception block
- 1D data instead of 2D images: 1D convolutions needed
- Adjusted the **kernel sizes**
		- EEG filters: 19, 25, 33, 39 → covering time windows of 0.3–0.6 seconds
		- Audio filters: 65, 81 → covering 1.0–1.2 seconds
- Removed the “pooling” branch

##### Avoid bias
The model might accidentally learn that, say, “Speaker 1” is usually on the left channel.  
To prevent this bias:
- They **duplicate** the data and **swap** the left/right audio inputs.
- They also **switch the labels** (which one is the attended speaker).

#### **Datasets**
- EventAAD dataset
- DTU
- KUL


#### **Evaluation procedure**

**split the data by trials and by subjects**, so the model never sees the same trial (or attended stimulus) during both training and testing

**Subject specific (SS)**: <font color="#f79646">8 fold cross validation</font>
- split one person's trials into 8 parts (7 training+validation, 1 testing). Repeat 8 times, then average test results to get accuracy score
- inside each training session (4:1 ratio): **Training set (80%)**, **Validation set (20%)**
- each trial is divided into smaller segments

**Subject-independent (SI)**: <font color="#f79646">LOSO: cross-trial leave one subject out cross validation</font>
	*problem*: In one dataset they used (the EventAAD dataset), all subjects heard the same exact sound sequences in the same order.
	**Extra careful split**:
		1. **Pick a test** *subject i*
		2. Split that subject i’s data into 8 folds (*1 for testing +  7 folds not used in this iteration)*
	    3. For the remaining subjects:
		    - Remove any trials where the attended stimulus also appears in subject i’s fold.
	         - From the rest (**Training set** (80%), **Validation set** (20%))
	    4. Train the model, test it on the held-out fold, and record performance
	    5. Repeat for all 8 folds
	![[image-2 1.png]]
**Result:**  
The model never gets to see the same attended stimulus in both training and testing, even if it’s from different people


#### Performance metrics
1) classification accuracy --> labels correspond
2) MESD

#### Hyperparameter choice and model training
- Loss function = mathematical function it tries to minimize 
		<font color="#8064a2">cross-entropy loss function</font>: standard loss for **classification** problems. Penalizes the model heavily when it’s confident but wrong.
    
- Optimizer: what algorithm model uses to update its parameters 
	    <font color="#8064a2">AdamW optimizer</font>: includes **weight decay** (=kind of regularization that keeps weights small to avoid overfitting), but it **decouples** it from the learning rate. That means you can control how much the model “forgets” large weights separately from how fast it learns. 

- Hyperparameters: parameters you have to set _before_ training (batch size, learning rate, dropout, etc.)
		<font color="#8064a2">random search</font>: try random combinations from a set of options and pick the best-performing one
		special case: h=0: removed first FC

- Training strategy: when to stop (where model stops learning and starts to overfit)
		<font color="#8064a2">validation-based early stopping rule</font>: While training, the model tracks both: Training loss & validation loss
		- if validation loss improves (goes down): **save model**
		- if does not improve for 5 epochs in a row: **stop**
		
- implementation: 
	- **PyTorch** → for deep learning models (AADNet and NSR)
    - **Scikit-learn** → for simpler, classical models:
	    - **RidgeCV** → used for the LSR (Linear Regression) model
	    - **CCA** → Canonical Correlation Analysis model  
	    - **LDA** → Linear Discriminant Analysis, for classification based on CCA features



#### **Results**
SS:
- Accuracy **improves with longer window lengths** (more EEG data per decision)
- outperforms other methods on most datasets and time windows. Exception: for short windows (≤10 s), linear better

SI:
measured **how much accuracy drops** when switching from subject-specific (SS) to subject-independent (SI). drop is **smallest for AADNet** compared to other methods, drop **increases with longer windows**

#### **Discussion**
- Even modest improvements in SI models matter — because they: Don’t require per-subject retraining, making them ready for real-world use. They use fixed hyperparameters across datasets, simplifying deployment.
        
- AADNet also achieved the lowest MESD (fastest reaction time) among SI models → good for real-time hearing devices, But: AADNet’s **short-window performance** still trails linear models → future work should improve responsiveness for shorter windows.

- Leave-one channel out strategy to learn if biologically meaningful spatial patterns were learned (language processing regions of the brain)



---
### Analogy training strategy: Think of Training Like Cooking

- **Loss function** = taste test (how bad the soup tastes)
- **Optimizer** = your spoon (how you adjust ingredients)
- **Hyperparameters** = your recipe settings (salt amount, cooking time)
- **Validation loss** = your friend’s feedback after each try
- **Early stopping** = you stop cooking once your friend says it’s perfect
- **Fine-tuning** = you take your favorite soup base and adapt it slightly for someone who prefers it spicier (different subject)

