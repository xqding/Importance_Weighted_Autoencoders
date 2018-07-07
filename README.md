# Importance Weighted Autoencoders
This is a PyTorch implementation of the importance weighted autoencoders (IWAE) proposed in 
[the paper](https://arxiv.org/pdf/1509.00519.pdf) by Yuri Burda, Roger Grosse, and Ruslan Salakhutdinov.
The implementation was tested on the MNIST dataset to replicate the result in the above paper.

- With one stochastic layer, the negative log-likelihood (estimated using k = 5000 as in the paper) on the test set is

 k | VAE   | IWAE 
---| ---   | ----
1  | 87.78 | 87.75
5  | 87.15 | 85.46
50 | 87.47 | 84.59

- With two stochastic layers, the negative log-likelihood on the test set is

 k | VAE   | IWAE 
---| ---   | ----
1  | 89.16 | 88.39
5  | 91.37 | 84.69
50 | 89.16 | 83.29

The result obtained here about IWAE agrees with the result from the original paper. As the number of samples (k) used to calcualte
ELBO increases, IWAE's performance (evaluated by an estimate of negative log-likelihood) improves. 
The values obtained here are not exactly equal to that in the original paper, because the optimization procesure used in this 
implementation is not the same as that in the paper.
We also note that the performance of VAE with two stochastic layers is worse than that of VAE with one stochatic layer, which does 
not agree with the original paper. 
This might be bacause we are using the same optimization procedure for VAE with both one and two stochastic layers and the optimization
procedure is not optimial for VAE with two stochastic layers.
