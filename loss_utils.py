import jax
import jax.numpy as jnp

import flax
import optax
import diffrax

from flax.training import common_utils

from func_utils import dot_product

def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=10)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics

def loss(model, params, batch, weight_decay=1e-4):
    """loss function used for training."""
    logits = model.apply(params, batch[0])
    loss = cross_entropy_loss(logits, batch[1])
    weight_l2 = dot_product(params, params)
    weight_penalty = weight_decay * 0.5 * weight_l2
    loss = loss + weight_penalty
    return loss, logits