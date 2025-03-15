# TARFLOW Implementation in `jax` .

Implementation of TARFLOW from [Normalizing Flows are Capable Generative Models](https://arxiv.org/pdf/2412.06329) using `jax` .

## Features  
- Efficient layer construction and forward pass using `jax.vmap` and `jax.lax.scan` for optimized compilation and execution.  
- Multi-device support for training, inference, and sampling.  
- Score-based denoising step.  
- Conditional modeling via class embeddings (for discrete labels) or adaptive layer normalization (for continuous variables, similar to DiT).  

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
