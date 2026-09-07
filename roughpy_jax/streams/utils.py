import jax
import jax.numpy as jnp


def _index_stream_batch(data: jax.Array, index) -> jax.Array:
    """Index the batch axes of stream data while preserving structural axes."""
    batch_index = index if isinstance(index, tuple) else (index,)

    # Put the structural axis immediately before the coefficient axis. These
    # trailing axes are then protected from the user's index, including when
    # JAX moves the result axes of separated advanced indices to the front.
    batch_first = jnp.moveaxis(data, 0, -2)
    selected = batch_first[batch_index + (slice(None), slice(None))]

    if 0 in selected.shape[:-2]:
        raise ValueError("batch index would produce an empty stream")

    return jnp.moveaxis(selected, -2, 0)
