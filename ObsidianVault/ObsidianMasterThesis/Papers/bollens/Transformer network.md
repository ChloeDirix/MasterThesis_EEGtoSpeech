Neural network architecture that uses **attention** to model relationships in data (like words in a sentence) more efficiently and effectively than older sequence models.

*difference with standard RNNs?*
1. RNNs processes data step by step, transformers use _attention (=Context)_ to figure out which parts of the input are most relevant when producing an output.
2. RNNs process sequences sequentially, transformers process entire sequences at once, making them much faster to train on modern hardware (GPUs/TPUs).
### Example: 
	input = sentence
	output = translated sentence
	token = word
### Components
- **Encoder** (left): input --> representation.
- **Decoder** (right): Representation --> output 

- **Self-Attention + Feed-Forward Layers**: The building blocks inside both encoder and decoder.
![[Transformer.png]]

##### Input embeddings & Positional Encoding
Since transformers don’t process tokens in order like RNNs, they need a way to represent the order of elements in a sequence. They do this with _positional encodings_ (extra vectors added to embeddings).

##### Multi-Head self-Attention Mechanism

**self-attention**:
	Each token looks at every other tokens and decides how much attention to give it in the new representation
**Multi-head self-attention**
	The model does it in parallel with several “heads,” each learning a different way to relate tokens (syntax semantics, long distance relations, etc.). Results from all heads are concatenated and mixed back together.
	
**How it works?**
	For each token embedding x, the model creates three vectors
	- **Query (Q)** – what am I looking for?
    - **Key (K)** – what information do I have?
    - **Value (V)** – the actual content I can pass along.

1. Attention Score between two tokens= how relevant one token is to another
$$
AttentionScore= Q.K^T
$$
2. Scores are scaled (so values don't blow up) and passed through a [[Softmax]] so they scale to 1 and the values are scaled
$$
Att(Q,K,V)= Softmax(\frac{Q.K^T}{\sqrt k})*V
$$
3. The weighted sum of all V's gives the new representation
$$
New Representation=Sum(att(Q,K,V))
$$

##### Feed Forward mechanism (FFN)
mini neural network applied to each token separately and identically, not mixing information like with attention

*Attention network*: mixes and relates information **linearly**
*FFN*: adds non-linearity and complexity, letting the model learn abstract features like
- Syntax roles (“this is likely a verb”),
- Semantic categories (“this token relates to animals”),
- Contextual refinements.

###### Intuitive explanation
attention: like gathering information from all your friends about a topic --> tells you what to focus on
FFN is like processing that information inside your own brain
- You expand your thoughts (projection up),
- Apply reasoning and interpretation (non-linear activation),
- Then condense your refined understanding (projection down).
	


##### Residual Connections
residual (skip) connection adds the input of a layer back to its output:
$$
Output=Layer(x) + x
$$
**goal:** helps preserve information and prevents gradients from vanishing!

##### Layer Normalization
Applied after residual connections.
→ Normalizes activations so they have mean 0 and variance 1 
→ makes training more stable and faster.