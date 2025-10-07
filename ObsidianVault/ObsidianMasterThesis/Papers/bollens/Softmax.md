 is a mathematical function that converts a vector of real numbers into a probability distribution, where each value is between 0 and 1 and the sum of all values equals 1

transforms raw scores (logits) into probabilities that can be interpreted as the likelihood of belonging to each class.

![[SoftMax.png]]

***Why we use it?***
--> Convert raw attention scores into normalized weights.
Without it, attention scores could be unbounded. Softmax ensures they form a probability distribution over tokens (so you can say “I pay 70% attention here, 20% there, 10% elsewhere”).