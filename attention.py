import functools as ft
import math
import warnings
from typing import Callable, Literal, Optional, Tuple, Union
import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.random as jr
from equinox.nn import Dropout, Linear, State, StateIndex
from jaxtyping import Array, Bool, Float, PRNGKeyArray, PyTree, jaxtyped
from beartype import beartype as typechecker

typecheck = jaxtyped(typechecker=typechecker)

@typecheck
def standard_attention(
    query_heads: Float[Array, "q_seq num_heads q_size"],
    key_heads: Float[Array, "kv_seq num_heads k_size"],
    value_heads: Float[Array, "kv_seq num_heads v_size"],
    num_heads: int,
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    inference: Optional[bool] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "#q_seq kv_seq"]] = None,
    *,
    keys: Optional[PRNGKeyArray] = None,
):
    attn_fn = ft.partial(dot_product_attention, dropout=dropout, inference=inference, scale_factor=scale_factor, attn_bias=attn_bias)
    in_axes = (1, 1, 1, 0 if mask is not None and mask.ndim == 3 else None)
    return jax.vmap(attn_fn, in_axes=in_axes, out_axes=1, axis_size=num_heads)(query_heads, key_heads, value_heads, mask, key=keys)

@typecheck
def dot_product_attention_weights(
    query: Float[Array, "q_seq qk_size"],
    key: Float[Array, "kv_seq qk_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "1 kv_seq"]] = None,
) -> Float[Array, "q_seq kv_seq"]:
    query, key = (query / scale_factor, key / scale_factor) if scale_factor else (query / math.sqrt(query.shape[-1]), key / math.sqrt(key.shape[-1]))
    logits = jnp.einsum("sd,Sd->sS", query, key)
    if attn_bias is not None:
        logits += jnp.broadcast_to(attn_bias, (query.shape[0], attn_bias.shape[-1]))
    if mask is not None:
        if mask.shape != logits.shape:
            raise ValueError(f"mask must have shape ({query.shape[0]}, {key.shape[0]}). Got {mask.shape}.")
        logits = jnp.where(mask, logits, jnp.finfo(logits.dtype).min)
    return jax.nn.softmax((logits - jnp.max(logits)).astype(jnp.float32), axis=-1).astype(query.dtype)

@typecheck
def dot_product_attention(
    query: Float[Array, "q_seq qk_size"],
    key_: Float[Array, "kv_seq qk_size"],
    value: Float[Array, "kv_seq v_size"],
    mask: Optional[Bool[Array, "q_seq kv_seq"]] = None,
    dropout: Optional[Dropout] = None,
    scale_factor: Optional[float] = None,
    attn_bias: Optional[Float[Array, "1 kv_seq"]] = None,
    *,
    key: Optional[PRNGKeyArray] = None,
    inference: Optional[bool] = None,
) -> Float[Array, "q_seq v_size"]:
    weights = dot_product_attention_weights(query, key_, mask, scale_factor, attn_bias=attn_bias)
    if dropout:
        weights = dropout(weights, key=key, inference=inference)
    return jnp.einsum("sS,Sd->sd", weights, value)

def vmapped_attention(
    query_heads: Float[Array, "seq_length query_multihead_dim qk_size"],
    key_heads: Float[Array, "seq_length qk_size"],
    value_heads: Float[Array, "seq_length v_size"],
    dropout: Optional[Dropout] = None,
    inference: Optional[bool] = None,
    mask: Optional[Float[Array, "q_seq kv_seq"]] = None,
    keys: Optional[PRNGKeyArray] = None,
):
    attn_fn = ft.partial(dot_product_attention, dropout=dropout, inference=inference, key=keys, mask=mask)
    return jax.vmap(lambda q, k, v: attn_fn(q, k, v), in_axes=(1, None, None), out_axes=1)(query_heads, key_heads, value_heads)

class MultiheadAttention(eqx.Module):
    query_proj: Linear
    key_proj: Linear
    value_proj: Linear
    output_proj: Linear
    dropout: Dropout
    autoregressive_index: StateIndex
    num_heads: int = eqx.field(static=True)
    query_size: int = eqx.field(static=True)
    key_size: int = eqx.field(static=True)
    value_size: int = eqx.field(static=True)
    output_size: int = eqx.field(static=True)
    state_length: Optional[int] = eqx.field(static=True)
    qk_size: int = eqx.field(static=True)
    vo_size: int = eqx.field(static=True)
    use_query_bias: bool = eqx.field(static=True)
    use_key_bias: bool = eqx.field(static=True)
    use_value_bias: bool = eqx.field(static=True)
    use_output_bias: bool = eqx.field(static=True)
    query_multihead_dim: int = eqx.field(static=True)
    kv_multihead_dim: int = eqx.field(static=True)
    kv_interpolation_mode: Literal["average", "repeat"] = eqx.field(static=True)
    scale_factor: Optional[float] = eqx.field(static=True)
    attn_bias: Float[Array, "1 q"]

    @typecheck
    def __init__(
        self,
        num_heads: int,
        query_size: int,
        *,
        key_size: Optional[int] = None,
        value_size: Optional[int] = None,
        output_size: Optional[int] = None,
        query_multihead_dim: Optional[int] = None,
        kv_multihead_dim: Optional[int] = None,
        state_length: Optional[int] = None,
        qk_size: Optional[int] = None,
        vo_size: Optional[int] = None,
        use_query_bias: bool = False,
        use_key_bias: bool = False,
        use_value_bias: bool = False,
        use_output_bias: bool = False,
        dropout_p: float = 0.0,
        inference: bool = False,
        kv_interpolation_mode: Literal["average", "repeat"] = "average",
        scale_factor: Optional[float] = None,
        attn_weight_bias: bool = False,
        key: PRNGKeyArray,
        **kwargs,
    ):
        qkey, kkey, vkey, okey = jr.split(key, 4)
        key_size = key_size or query_size
        value_size = value_size or query_size
        qk_size = qk_size or query_size // num_heads
        vo_size = vo_size or query_size // num_heads
        output_size = output_size or query_size
        query_proj_out_size = qk_size * (query_multihead_dim or num_heads)
        key_proj_out_size = qk_size * (kv_multihead_dim or num_heads)
        value_proj_out_size = vo_size * (kv_multihead_dim or num_heads)
        self.query_proj = Linear(query_size, query_proj_out_size, use_bias=use_query_bias, key=qkey)
        self.key_proj = Linear(key_size, key_proj_out_size, use_bias=use_key_bias, key=kkey)
        self.value_proj = Linear(value_size, value_proj_out_size, use_bias=use_value_bias, key=vkey)
        self.output_proj = Linear(vo_size * num_heads, output_size, use_bias=use_output_bias, key=okey)
        self.dropout = Dropout(dropout_p, inference=inference)
        self.autoregressive_index = StateIndex(self._make_autoregressive_cache(state_length, num_heads, qk_size, vo_size, kv_multihead_dim))
        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.qk_size = qk_size
        self.vo_size = vo_size
        self.use_query_bias = use_query_bias
        self.use_key_bias = use_key_bias
        self.use_value_bias = use_value_bias
        self.use_output_bias = use_output_bias
        self.state_length = state_length
        self.kv_multihead_dim = kv_multihead_dim or num_heads
        self.query_multihead_dim = query_multihead_dim or num_heads
        self.kv_interpolation_mode = kv_interpolation_mode
        self.scale_factor = scale_factor
        self.attn_bias = jnp.zeros((1, state_length)) if attn_weight_bias else None

    def _make_autoregressive_cache(self, state_length, num_heads, qk_size, vo_size, kv_multihead_dim):
        if state_length is None:
            raise ValueError("Cannot use autoregressive decoding without specifying `state_length`.")
        key_shape = (state_length, num_heads, qk_size) if kv_multihead_dim else (state_length, qk_size)
        value_shape = (state_length, num_heads, vo_size) if kv_multihead_dim else (state_length, vo_size)
        dtype = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
        return jnp.empty(key_shape), jnp.empty(value_shape), jnp.zeros((), dtype)

    @typecheck
    def __call__(
        self,
        query: Float[Array, "q_seq q_size"],
        key_: Float[Array, "kv_seq k_size"],
        value: Float[Array, "kv_seq v_size"],
        mask: Union[
            None,
            Bool[Array, "q_seq kv_seq"],
            Bool[Array, "num_heads q_seq kv_seq"],
            Literal["causal"],
        ] = None,
        state: Optional[State] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
        inference: Optional[bool] = None,
        deterministic: Optional[bool] = None,
        process_heads: Optional[Callable] = None,
    ) -> Union[Float[Array, "q_seq o_size"], Tuple[Float[Array, "q_seq o_size"], State]]:
        if deterministic is not None:
            inference = deterministic
            warnings.warn("`deterministic` is deprecated in favour of `inference`.")
        query_seq_length, _ = query.shape
        kv_seq_length, _ = key_.shape
        if kv_seq_length != value.shape[0]:
            raise ValueError("key and value must have the same sequence length.")
        query_heads = self._project(self.query_proj, self.query_multihead_dim, query)
        key_heads = self._project(self.key_proj, self.kv_multihead_dim, key_)
        value_heads = self._project(self.value_proj, self.kv_multihead_dim, value)
        if process_heads:
            shapes = (query_heads.shape, key_heads.shape, value_heads.shape)
            query_heads, key_heads, value_heads = process_heads(query_heads, key_heads, value_heads)
            if (query_heads.shape, key_heads.shape, value_heads.shape) != shapes:
                raise ValueError("process_heads must not change the shape of the heads.")
        causal_mask_offset = 0
        if state:
            key_state, value_state, index = state.get(self.autoregressive_index)
            key_state = lax.dynamic_update_slice_in_dim(key_state, key_heads, index, axis=0)
            value_state = lax.dynamic_update_slice_in_dim(value_state, value_heads, index, axis=0)
            causal_mask_offset = index
            index = (index + kv_seq_length) % self.state_length
            state = state.set(self.autoregressive_index, (key_state, value_state, index))
            key_heads, value_heads = key_state, value_state
            kv_seq_length = self.state_length
        if mask == "causal":
            mask = jnp.arange(kv_seq_length)[None, :] <= jnp.arange(query_seq_length)[:, None] + causal_mask_offset
        if state:
            unwritten_mask = jnp.arange(self.state_length) < index
            mask = unwritten_mask if mask is None else mask & unwritten_mask
        keys = None if key is None else jr.split(key, self.num_heads)
        attn = standard_attention(query_heads, key_heads, value_heads, self.num_heads, mask, self.dropout, inference, self.attn_bias, keys=keys)
        out = jax.vmap(self.output_proj)(attn.reshape(query_seq_length, -1))
        return out if state is None else (out, state)

    @typecheck
    def _project(self, proj: PyTree, multihead: int, x: Array) -> Array:
        seq_length, _ = x.shape
        projection = jax.vmap(proj)(x)
        return projection.reshape(seq_length, multihead, -1) if multihead else projection

def self_attention(
    num_heads: int,
    size: int,
    *,
    multiquery: bool = False,
    state_length: Optional[int] = None,
    scale_factor: Optional[float] = None,
    attn_weight_bias: bool = False,
    key: PRNGKeyArray,
):
    return MultiheadAttention(
        num_heads=num_heads,
        query_size=size,
        state_length=state_length,
        key_multihead=not multiquery,
        value_multihead=not multiquery,
        scale_factor=scale_factor,
        attn_weight_bias=attn_weight_bias,
        key=key,
    )
