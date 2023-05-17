# AdamK optimizer

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import call
from jax import lax, vmap

from func_utils import hvp, dot_product, fisher_kernel_func

def custom_jit(fun):
    return jax.jit(fun, static_argnums=(0,))

def adam_init(params, learning_rate=1, beta1=0.9, beta2=0.99, eps=1e-8):

    # Initializing state dict, lambd and weight_decay should be incorporated later
    state = {}
    state['learning_rate'] = learning_rate
    state['beta1'] = beta1
    state['beta2'] = beta2
    state['eps'] = eps
    state['m'] = jax.tree_map(jnp.zeros_like, params)
    state['v'] = jax.tree_map(jnp.zeros_like, params)
    state['damps'] = jax.tree_map(jnp.zeros_like, params)
    state['t'] = 0
    return state

@jax.jit
def get_adam_vanilla(params, grads, state):
    
    # Jitted function to update the first moment estimate (m) for each parameter
    @jax.jit
    def update_m(m, grad):
        return state['beta1'] * m + (1.0 - state['beta1']) * grad

    # Jitted function to update the second moment estimate (v) for each parameter
    @jax.jit
    def update_v(v, grad):
        return state['beta2'] * v + (1.0 - state['beta2']) * jnp.square(grad)

    @jax.jit
    def update_mv(m, v):
        return m / (jnp.sqrt(v) + state['eps'])

    state['m'] = jax.tree_map(update_m, state['m'], grads)
    state['v'] = jax.tree_map(update_v, state['v'], grads)

    adams = jax.tree_map(update_mv, state['m'], state['v'])

    return adams

@custom_jit
def get_adam(loss, params, grads, state, batch, rho=0.05):
    
    # Jitted function to update the first moment estimate (m) for each parameter
    @jax.jit
    def update_m(m, grad):
        return state['beta1'] * m + (1.0 - state['beta1']) * grad

    # Jitted function to update the second moment estimate (v) for each parameter
    @jax.jit
    def update_v(v, grad):
        return state['beta2'] * v + (1.0 - state['beta2']) * jnp.square(grad)

    @jax.jit
    def update_mv(m, v):
        return m / (jnp.sqrt(v) + state['eps'])

    grads_norm = jnp.sqrt(dot_product(grads, grads))

    # Compute the perturbed parameters
    @jax.jit
    def perturb_params(param, grads):
        return param + rho * grads / grads_norm

    perturbed_params = jax.tree_map(perturb_params, params, grads)

    # Compute gradients at the perturbed parameters
    perturbed_grads = jax.grad(loss)(perturbed_params, batch)

    # Update the Adam state using the perturbed gradients
    state['m'] = jax.tree_map(update_m, state['m'], perturbed_grads)
    state['v'] = jax.tree_map(update_v, state['v'], perturbed_grads)

    # Compute the new Adam update vector using the perturbed gradients
    new_adams = jax.tree_map(update_mv, state['m'], state['v'])

    return new_adams

@custom_jit
def get_optim(model_fn, params, batch, grads, adams, damps, lambd, weight_decay):

    # Compute the JVP for a batch of data
    logits, jvp_adam = jax.jvp(model_fn, (params, batch[0]), (adams, jnp.zeros_like(batch[0])))
    logits, jvp_damp = jax.jvp(model_fn, (params, batch[0]), (damps, jnp.zeros_like(batch[0])))

    get_fisher_kernel = jax.vmap(fisher_kernel_func)
    fisher_kernel = get_fisher_kernel(logits)

    def vMv_product(vector1, vector2, matrix):
        return jnp.einsum('bi, bij, bj -> b', vector1, matrix, vector2)

    adam_F_adam = jnp.mean(vMv_product(jvp_adam, jvp_adam, fisher_kernel))
    adam_F_damp = jnp.mean(vMv_product(jvp_adam, jvp_damp, fisher_kernel))
    damp_F_damp = jnp.mean(vMv_product(jvp_damp, jvp_damp, fisher_kernel))

    mat11 = (adam_F_adam + (lambd + weight_decay) * dot_product(adams, adams)) / 100
    mat12 = adam_F_damp + (lambd + weight_decay) * dot_product(adams, damps)
    mat21 = (adam_F_damp + (lambd + weight_decay) * dot_product(adams, damps)) / 100
    mat22 = damp_F_damp + (lambd + weight_decay) * dot_product(damps, damps)

    # # Compute FIM vector product with adam vector \Delta and damping vector \delta
    # hvp_adam = hvp(loss, params, batch, adams)
    # hvp_damp = hvp(loss, params, batch, damps)

    # call(lambda x: print(x), optim)

    # # Construct 2 by 2 matrix for the optimal vector
    # mat11 = dot_product(adams, hvp_adam) + (lambd + weight_decay) * dot_product(adams, adams)
    # mat12 = dot_product(adams, hvp_damp) + (lambd + weight_decay) * dot_product(adams, damps)
    # mat21 = mat12
    # mat22 = dot_product(damps, hvp_damp) + (lambd + weight_decay) * dot_product(damps, damps)

    mat = jnp.array([[mat11, mat12], [mat21, mat22]])

    # Add small positive value to the diagonal for better numerical stability
    alpha = 1e-5
    mat += jax.numpy.eye(2) * alpha

    condition_number = jax.numpy.linalg.cond(mat)

    # Function to handle the ill-conditioned or NaN condition_number case
    def ill_conditioned_case(_):
        return jax.numpy.array([[0.1], [0.01]])

    # Function to handle the well-conditioned case
    def well_conditioned_case(_):
        vec1 = dot_product(grads, adams)
        vec2 = dot_product(grads, damps)
        vec = jax.numpy.array([[vec1], [vec2]])
        optim = jax.numpy.linalg.solve(mat, vec)

        # Function to handle the NaN values in optim vector case
        def nan_values_case(_):
            return jax.numpy.array([[0.1], [-0.01]])

        # Function to handle the non-NaN values in optim vector case
        def non_nan_values_case(_):
            return optim

        # Use lax.cond to handle control flow with tracer values within well_conditioned_case
        return lax.cond(
            jax.numpy.isnan(optim).any(),
            nan_values_case,
            non_nan_values_case,
            None  # This argument is not used by either branch, so we pass None
        )

    # Use lax.cond to handle control flow with tracer values for condition_number
    optim = lax.cond(
        (condition_number > 1e8) | jax.numpy.isnan(condition_number),
        ill_conditioned_case,
        well_conditioned_case,
        None  # This argument is not used by either branch, so we pass None
    )

    # Apply constraints to optim
    optim = jnp.stack([jnp.clip(optim[0], -0.5, 1.5), jnp.clip(optim[1], -0.15, 0.7)])

    call(lambda x: print(x), optim)

    return optim

@custom_jit
def damp_update(model_fn, params, batch, grads, state, lambd, weight_decay):

    initial_learning_rate = state['learning_rate']
    final_learning_rate = 1.0
    total_steps = 200  # Set this to the number of total steps you want to take

    # Calculate the current learning rate with linear decay
    current_learning_rate = initial_learning_rate + (final_learning_rate - initial_learning_rate) * (state['t'] / total_steps)

    state['t'] += 1
    # Calculate the time-dependent learning rate
    lr_t = current_learning_rate * jnp.sqrt(1.0 - jnp.power(state['beta2'], state['t'])) / (1.0 - jnp.power(state['beta1'], state['t']))

    adams = get_adam_vanilla(params, grads, state)

    @jax.jit
    def update_params(param, damps):
        return param - lr_t * damps

    damps = state['damps']
    # Compute optimum \alpha and \mu for damp update
    optim = get_optim(model_fn, params, batch, grads, adams, damps, lambd, weight_decay)

    # Jitted function using optim vector to compute damp update
    @jax.jit
    def get_damps(adams, damps):
        return adams * optim[0] / 100 + damps * optim[1]

    damps = jax.tree_map(get_damps, adams, damps)

    # Compute the norm of the 'damps' variable
    damps_norm = jnp.sqrt(sum(jnp.sum(jnp.square(damp)) for damp in jax.tree_leaves(damps)))

    # Set the norm constraint limit
    norm_constraint = 1e+5
    
    @jax.jit
    def scale_damps(_):
        return jax.tree_map(lambda x: x * (norm_constraint / damps_norm), damps)

    @jax.jit
    def identity_damps(_):
        return damps

    # Apply the norm constraint if the norm of 'damps' is greater than the limit
    damps = lax.cond(damps_norm > norm_constraint, scale_damps, identity_damps, None)

    state['damps'] = damps

    params = jax.tree_map(update_params, params, damps)

    return params, state

@jax.jit
def adam_update(params, grads, state):
    initial_learning_rate = state['learning_rate']
    final_learning_rate = 1.0
    total_steps = 500  # Set this to the number of total steps you want to take

    # Calculate the current learning rate with linear decay
    current_learning_rate = initial_learning_rate + (final_learning_rate - initial_learning_rate) * (state['t'] / total_steps)

    state['t'] += 1
    # Calculate the time-dependent learning rate
    lr_t = current_learning_rate * jnp.sqrt(1.0 - jnp.power(state['beta2'], state['t'])) / (1.0 - jnp.power(state['beta1'], state['t']))

    adams = get_adam_vanilla(params, grads, state)

    @jax.jit
    def update_params(param, damps):
        return param - lr_t * damps

    damps = jax.tree_map(lambda x: 1e-3 * x, adams)
    params = jax.tree_map(update_params, params, damps)

    return params, state  

@custom_jit
def optimize_AdamK(model_fn, params, batch, grads, state, lambd, weight_decay):
    return damp_update(model_fn, params, batch, grads, state, lambd, weight_decay)

@jax.jit
def optimize_Adam(params, grads, state):
    return adam_update(params, grads, state)

