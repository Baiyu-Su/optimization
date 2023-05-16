import numpy as np

import jax
import jax.numpy as jnp
from jax import grad

from flax.training import common_utils

# @title hvp and dot product

def hvp(loss, params, batch, v):
    """Computes the hessian vector product Hv.

    This implementation uses forward-over-reverse mode for computing the hvp.

    Args:
      loss: function computing the loss with signature
        loss(params, batch).
      model: the model object.
      params: pytree for the parameters of the model.
      batch:  A batch of data. Any format is fine as long as it is a valid input
        to loss(model, params, batch).
      v: pytree of the same structure as params.

    Returns:
      hvp: array of shape [num_params] equal to Hv where H is the hessian.
    """

    def wrapped_loss_fn(params_for_diff):
        return loss(params_for_diff, batch)

    return jax.jvp(jax.grad(wrapped_loss_fn), [params], [v])[1]

def dot_product(v1, v2):
    
    """Computes dot product of two pytrees with same structure and dimension.

    Args:
        v1, v2: pytree1 and pytree2.

    Returns:
        result: scalar numpy array.
    """
    # Element-wise multiplication of corresponding leaves
    product_tree = jax.tree_map(lambda x, y: x * y, v1, v2)
    leaves = jax.tree_util.tree_leaves(product_tree)

    # Flatten the tree into a list of leaves and sum them all
    result = jnp.sum(jnp.concatenate([jnp.asarray(leaf).flatten() for leaf in leaves]))
    
    return result

def fisher_kernel_func(logits):
    diagonal = jnp.diag(logits)
    rankone = jnp.outer(logits, logits)

    return diagonal - rankone
