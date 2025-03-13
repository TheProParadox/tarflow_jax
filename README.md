Implementation of TARFLOW from Normalizing Flows are Capable Generative Models using jax and equinox.

Features:
Efficient layer construction and forward pass using jax.vmap and jax.lax.scan for optimized compilation and execution.
Multi-device support for training, inference, and sampling.
Score-based denoising step.
Conditional modeling via class embeddings (for discrete labels) or adaptive layer normalization (for continuous variables, similar to DiT).
To-Do:
 Implement guidance.
 Add denoising.
 Support mixed precision.
 Implement Exponential Moving Average (EMA).
 Integrate AdaLayerNorm.
 Enable class embeddings.
 Add hyperparameter and model saving.
 Use uniform noise for dequantization.
<!-- Notes: - The paper presents a promising generative modeling approach, but training is more challenging than implied. - Success depends on implementation details like attention mechanisms, requiring EMA and gradient clipping for stability. - The provided hyperparameters did not yield strong results in our tests. - The choice of quantization method remains unclear, as it affects both sample quality and model log-likelihood. - Training demands substantial compute resources. -->






