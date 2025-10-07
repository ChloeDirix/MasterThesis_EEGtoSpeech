
Both **ReLU** and **GELU** are _activation functions_. They’re the non-linear part of the FFN that allows the network to learn complex patterns.
![[GELU_RELU.png]]
.
	https://www.researchgate.net/figure/Comparison-of-the-ReLu-and-GeLu-activation-functions-ReLu-is-simpler-to-compute-but_fig3_370116538
### ReLU 
--> ON/OFF
Rectified Linear Unit
$$
ReLU(x)=max⁡(0,x)
$$
- If x>0: output is x.
- If x≤0: output is 0
    
**Pros**:
- Simple, cheap to compute.
- Avoids the “vanishing gradient” problem common in older activations (like sigmoid).
**Cons**:
- "Dead neurons": once a neuron outputs 0 (for negative inputs), it may never recover during training.

### GELU
--> Negatives are softened and positives are mostly kept
Gaussian Error Linear Unit
$$
GELU(x) = x.\phi(x)
≈0.5x(1+tanh(2/π​(x+0.044715x3)))
$$
- phi(x)= is the probability that a Gaussian random variable is less then x

**Pros**:
- Keeps some info from negative inputs (rather than throwing it away).
- Empirically works better for large language models (BERT, GPT, etc. all use GELU).