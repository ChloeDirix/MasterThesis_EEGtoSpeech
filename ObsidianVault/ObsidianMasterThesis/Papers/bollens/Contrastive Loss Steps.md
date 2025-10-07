
**Step 1: Encode into vectors**
	EEG encoder → vector zEi
	Speech encoder → vector zSi
		
**Step 2: Similarity Score**
	For each pair, compute dot product,  result is a KxK matrix with the true pairs on the diagonal
		  $$
		 sim(E_i,S_i) = <Si,Ei> = Ei . Si
		 $$
**Step 3: Softmax over candidates**
	Compare each Ei to all K  (1 true + K−1 negatives).	![[SoftMax_Over_Candidates.png]]
	==e^t = learnable scaling factor==
	
**Step 4: Cross-Entropy Loss**
	maximizes the probability of the correct pair.
	--> compute it symmetrically: both **EEG→Stimulus** and **Stimulus→EEG**.
	--> Final loss = average
		$$
		L_{E->S​}=-∑_{i=1}^K log({p_{Si}})​
		$$
$$
		L_{S→E​}=-∑_{i=1}^K log({p_{Ei}})​
		$$
		$$
		L_{CLIP}= (L_{E->S}+ L_{S->E})/2
		$$
	The model gets rewarded when true pairs are the closest and penalized otherwise