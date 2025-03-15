# TARFLOW Implementation in `jax` .

Implementation of TARFLOW from [Normalizing Flows are Capable Generative Models](https://arxiv.org/pdf/2412.06329) using `jax` .

## Features  
- Efficient layer construction and forward pass using `jax.vmap` and `jax.lax.scan` for optimized compilation and execution.  
- Multi-device support for training, inference, and sampling.  
- Score-based denoising step.  
- Conditional modeling via class embeddings (for discrete labels) or adaptive layer normalization (for continuous variables, similar to DiT).  

## Description of attention.py
- Complete multi-head attention mechanism with flexible query/key/value dimensions, customizable head counts, and optional projection biases
- Performance optimizations including autoregressive caching, JAX vectorization, and scale factor customization
- Comprehensive masking support with causal masks, custom attention biases, and flexible masking patterns
- Type-safe implementation with runtime shape validation and extensive annotations using jaxtyping and beartype
- Integrated training features like dropout regularization and state management for sequential processing

## Description of transformer_flow.py
- Implemented a Transformer-based normalizing flow model using Equinox and JAX, designed for image processing tasks with patch-based autoregression.
- Integrated multihead self-attention mechanisms to model complex dependencies between image patches, enabling structured and efficient learning.
- Designed a flexible conditioning framework supporting label-based, embedding-based, and layer normalization-based conditioning for improved adaptability.
- Optimized gradient accumulation and sharding strategies leveraging jax.sharding, optax, and Equinox filtering to efficiently distribute computations across multiple devices.
- Developed a robust training pipeline including dataset processing, noise augmentation, exponential moving average (EMA) updates, and loss tracking for stable convergence.

## Description of 
## To-Do  
- [ ] Implement guidance.  
- [x] Add denoising.  
- [x] Support mixed precision.  
- [x] Implement Exponential Moving Average (EMA).  
- [x] Integrate AdaLayerNorm.  
- [x] Enable class embeddings.  
- [x] Use uniform noise for dequantization.  

## Notes  
- The paper presents a promising generative modeling approach, but training is more challenging than implied.  
  - Success depends on implementation details like attention mechanisms, requiring EMA and gradient clipping for stability.  
  - The provided hyperparameters did not yield strong results in my tests.  
- Training demands substantial compute resources.  
