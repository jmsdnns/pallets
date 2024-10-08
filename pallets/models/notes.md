## On Hot Encoding (categorical classification) in the learning problem.

While you can modify a Variational Autoencoder (VAE) by applying an argmax across the 
color channel to automatically convert the output probabilities into a one-hot encoded vector
where only the maximum probability gets a value of 1, it's important to note that this operation 
is non-differentiable, which means it can't be used directly during the training phase if gradients 
need to be computed.

To implement this in a way that's compatible with training, you can use a "Gumbel-Softmax" 
trick or a "Straight-Through Gumbel-Softmax Estimator". These methods provide a 
differentiable approximation to the argmax function, allowing the network to learn 
through backpropagation while effectively producing one-hot encoded outputs during inference.

Here's a simplified explanation of how you might implement this:

Gumbel-Softmax Layer: At the output of your decoder, instead of using a standard softmax, 
you apply a Gumbel-Softmax operation. This operation adds Gumbel noise to the logits and 
then applies a softmax. The temperature parameter of the Gumbel-Softmax controls how close 
the output is to a one-hot vector. During training, you keep the temperature high enough 
to maintain differentiability. During inference, you can reduce the temperature to make 
the output more like a one-hot vector.

Straight-Through Estimator: During the forward pass, you use the Gumbel-Softmax to 
generate a soft approximation to the one-hot vector. During the backward pass (for gradient computation), 
you use a straight-through estimator, which essentially treats the Gumbel-Softmax as an argmax operation. 
This allows the gradients to flow through the non-differentiable argmax operation.

Code Example:

```python
import torch
import torch.nn.functional as F

def gumbel_softmax(logits, temperature=1.0, hard=False):
    # Draw samples from the Gumbel distribution
    gumbels = -torch.empty_like(logits).exponential_().log()
    gumbels = (logits + gumbels) / temperature
    y_soft = gumbels.softmax(dim=-1)

    if hard:
        # Create a straight-through estimator for the backward pass
        index = y_soft.max(dim=-1, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(-1, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft

    return ret

# Example usage in a VAE decoder
class VAE_Decoder(nn.Module):
    # ... define your layers ...

    def forward(self, z):
        # ... pass through layers ...
        logits = self.output_layer(z)  # Assume this is the output layer before applying Gumbel-Softmax
        return gumbel_softmax(logits, temperature=0.5, hard=True)  # Set hard=True for inference
```

In this example, gumbel_softmax is a function that applies the Gumbel-Softmax operation. 
You can adjust the temperature parameter based on your needs (lower temperatures make the output 
closer to one-hot). The hard parameter controls whether to use the straight-through estimator. 
During training, you might set hard=False and  use a higher temperature. For inference, set hard=True 
and use a lower temperature to get a one-hot encoded output.

### Reference

["Categorical Reparameterization with Gumbel-Softmax" Eric Jang, Shixiang Gu, and Ben Poole, 2022](https://openreview.net/forum?id=rkE3y85ee)
This paper introduces the Gumbel-Softmax distribution as a continuous distribution over the 
simplex that can approximate samples from a categorical distribution. The paper demonstrates 
its utility in learning discrete latent variable models.

["The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables" Chris J. Maddison, Andriy Mnih, and Yee Whye Teh, 2022](https://openreview.net/forum?id=S1jE5L5gl)
This paper independently introduces a similar concept known as the Concrete distribution, which is essentially the same 
as the Gumbel-Softmax distribution.

[Discrete Variational Autoencoders, Vae 2016](https://arxiv.org/abs/1609.02200)
Early work using a discrete latent space augmented with auxiliary continuous variables, a hierarchical posterior, and a RBM prior. 
Substantially different than ^^^ and also a difficult paper to read that would be very hard to reproduce.

[Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients ´
through stochastic neurons for conditional computation.](https://arxiv.org/abs/1308.3432)
