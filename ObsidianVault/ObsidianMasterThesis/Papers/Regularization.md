= introducing additional information (penalty” for complexity) to solve any ill-posed estimation problems and prevent overfitting

When you train a model, you’re estimating a bunch of weights (βs). EEG is noisy and **high-dimensional** (many electrodes, many time lags, many speech features)
==>**overfit**  to random noise instead of learning the true relationship.

Types
- L2 ridge
- L1 lasso
- elastic net
- dropout


| Model type               | Regularization examples                                         | Purpose                                                |
| ------------------------ | --------------------------------------------------------------- | ------------------------------------------------------ |
| **Linear / TRF**         | Ridge (L2), Lasso (L1)                                          | Prevent overfitting, stabilize weights                 |
| **Transformers**         | Dropout, weight decay (L2), early stopping, layer normalization | Prevent overfitting and help generalization            |
| **Contrastive learning** | Temperature scaling, data augmentation, embedding normalization | Prevent trivial solutions (e.g., identical embeddings) |