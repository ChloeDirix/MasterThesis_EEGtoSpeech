### **Convolutional layer:**
= Front-end embedding step: turns raw signal into sequence of tokens
<span style="background:rgba(240, 200, 0, 0.2)">-- captures temporal patterns across short speech segments</span>

1. <u>convolutional layer</u> (64 filters, kernel size = 64) 
	→ each filter looks at 64 time points at once 
	→ captures temporal patterns across short speech segments
	
2. <u>Layer normalization:</u> makes training more stable by normalizing across the layer’s features (stabilizes activations)

3. <u>GELU activation:</u> [[GELU & RELU]] a smooth, nonlinear function


*In parallel: skip connection* [[Transformer network#Residual Connections]]

---
### **Transformer based block**
<span style="background:rgba(240, 200, 0, 0.2)">capture long-range temporal dependencies across the EEG window</span>

#### 1. **multi-head self attention (h=8)**
lets each time point in the EEG look at all other time points and decide which are important.

##### The process:
1. Compute Q, K and V's.
2. AttentionScore (dot product) measures similarity between time points
3. Scale and Apply **Softmax** → get attention weights
4. Multiply by V → produce weighted combinations			$$
			Att(Q,K,V) = Softmax(\frac{QK^T}{\sqrt k})V
			$$

##### multi-head attention (h=8):
Tokens are split into 8 chunks and each chunk goes through its own attention. Outputs are concatenated.

		
*in parallel: skip connection* [[Transformer network#Residual Connections]]
		 
#### 2. **Feed forward (MLP)**
Expands it to a higher dimension (4×) for each token and applies GELU, then compresses back
--> capture more complex transformations
