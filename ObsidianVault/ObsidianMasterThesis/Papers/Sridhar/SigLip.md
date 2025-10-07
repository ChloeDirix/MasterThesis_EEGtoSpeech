**traditional methods**:
infoNCE & Triplet loss depend on predefined positive and negative samples

**CLIP**: 
Compares all pairs within a batch, projecting multimodal inputs in a shared embedding space.
A softmax-based loss aligns matching pairs while separating mismatches, but performance depends heavily on large batch sizes

**SigLip**
variant of CLIP that uses a sigmoid loss instead of a softmax loss. Task becomes binary classification
--> works better for smaller batches
![[CLIP.png]]
