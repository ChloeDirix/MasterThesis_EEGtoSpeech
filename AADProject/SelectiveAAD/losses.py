"""
cross-entropy over candidates

Given:
logits = [B, K] from model
target = [B] with integers in [0..K-1] (attended_index)
"""

#loss = F.cross_entropy(logits / temperature, target)