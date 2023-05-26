# AdamK optimizer

import jax
import jax.numpy as jnp
from jax.experimental.host_callback import call
from jax import lax, vmap

from func_utils import dot_product, fisher_kernel_func

def custom_jit(fun):
    return jax.jit(fun, static_argnums=(0,))

def adam_init(params, learning_rate=1, beta1=0.9, beta2=0.999, beta3=0.7, eps=1e-7):
    # Initializing state dict

    state = {}
    state['learning_rate'] = learning_rate
    state['beta1'] = beta1
    state['beta2'] = beta2
    state['beta3'] = beta3
    state['eps'] = eps
    state['m'] = jax.tree_map(jnp.zeros_like, params)
    state['v'] = jax.tree_map(jnp.zeros_like, params)
    state['damps'] = jax.tree_map(jnp.zeros_like, params)
    state['adams'] = jax.tree_map(jnp.zeros_like, params)
    state['product'] = 0.0
    state['t'] = 0
    return state

@jax.jit
def get_adam(step, grads, state):
    
    # Jitted function to update the first moment estimate (m) for each parameter with bias correction
    @jax.jit
    def update_m(m, grad):
        return (state['beta1'] * m + (1.0 - state['beta1']) * grad)

    # Jitted function to update the second moment estimate (v) for each parameter with bias correction
    @jax.jit
    def update_v(v, grad):
        return (state['beta2'] * v + (1.0 - state['beta2']) * jnp.square(grad))
    
    def update_mv(m, v):
        m_hat = m / (1 - state['beta1'] ** (step + 1))
        v_hat = v / (1 - state['beta2'] ** (step + 1))
        return m_hat / (jnp.sqrt(v_hat) + state['eps'])

    state['m'] = jax.tree_map(update_m, state['m'], grads)
    state['v'] = jax.tree_map(update_v, state['v'], grads)

    adams = jax.tree_map(update_mv, state['m'], state['v'])

    return adams, state

@custom_jit
def get_optim_bsc(model_fn, params, step, batch, grads, adams, state, lambd, weight_decay):

    adam_F_adam = get_adam_F_adam(model_fn, params, batch, adams)
    mat11 = (adam_F_adam + (lambd + weight_decay) * dot_product(adams, adams))
    
    optim = jnp.array([dot_product(grads, adams)/mat11, 0.0])
    
    #call(lambda x: print(x), optim[0])
    return optim

@custom_jit
def get_optim_wpm(model_fn, params, step, batch, grads, adams, state, lambd, weight_decay):

    adam_F_adam = get_adam_F_adam(model_fn, params, batch, adams)
    get_adam_F_adam_grad = jax.grad(get_adam_F_adam, argnums=(1, 3))
    params_grad, adams_grad = get_adam_F_adam_grad(model_fn, params, batch, adams)

    corr = -dot_product(params_grad, state['damps']) + dot_product(adams_grad, adams-state['adams'])
    state['product'] = state['beta3'] * (state['product'] + corr) + (1 - state['beta3']) * adam_F_adam

    mat11 = state['product'] / (1 - state['beta3'] ** (step + 1)) + (lambd + weight_decay) * dot_product(adams, adams)

    optim = jnp.array([dot_product(grads, adams)/mat11, 0.0])
    #call(lambda x: print(x), optim[0])
    return optim

@custom_jit
def get_adam_F_adam(model_fn, params, batch, adams):
    # Compute the JVP for a batch of data
    logits, jvp_adam = jax.jvp(model_fn, (params, batch[0]), (adams, jnp.zeros_like(batch[0])))
    get_fisher_kernel = jax.vmap(fisher_kernel_func)
    fisher_kernel = get_fisher_kernel(logits)
    
    @jax.jit
    def vMv_product(vector1, vector2, matrix):
        return jnp.einsum('bi, bij, bj -> b', vector1, matrix, vector2)

    adam_F_adam = jnp.mean(vMv_product(jvp_adam, jvp_adam, fisher_kernel))

    return adam_F_adam

@custom_jit
def damp_update(params, step, batch, grads, state, lambd, weight_decay, model_fn, get_optim):

    lr_t = state['learning_rate']
    adams, state = get_adam(step, grads, state)

    # Compute optimum \alpha and \mu for damp update
    optim = get_optim(model_fn, params, step, batch, grads, adams, state, lambd, weight_decay)

    # Jitted function using optim vector to compute damp update
    @jax.jit
    def get_damps(adams, damps):
        return lr_t * (adams * optim[0] + damps * optim[1])

    damps = jax.tree_map(get_damps, adams, damps)

    # Compute the norm of the 'damps' variable
    damps_norm = jnp.sqrt(dot_product(damps, damps))
    # call(lambda x: print(x), damps_norm)

    # Set the norm constraint limit
    norm_constraint = 20
    @jax.jit
    def scale_damps(_):
        return jax.tree_map(lambda x: x * (norm_constraint / damps_norm), damps)
    @jax.jit
    def identity_damps(_):
        return damps
    # Apply the norm constraint if the norm of 'damps' is greater than the limit
    state['damps'] = lax.cond(damps_norm > norm_constraint, scale_damps, identity_damps, None)
    state['adams'] = adams

    params = jax.tree_map(lambda x, y: x-y, params, state['damps'])

    return params, state

@jax.jit
def adam_update(params, step, grads, state):
    # Compute adam vector and update parameters

    lr_t = state['learning_rate']
    adams, state = get_adam(step, grads, state)

    @jax.jit
    def update_params(param, damps):
        return param - lr_t * damps

    damps = jax.tree_map(lambda x: 1e-3 * x, adams)
    params = jax.tree_map(update_params, params, damps)

    return params, state  

@custom_jit
def optimize_AdamK_bsc(model_fn, params, step, batch, grads, state, lambd, weight_decay):
    return damp_update(params, step, batch, grads, state, lambd, weight_decay, model_fn, get_optim_bsc)

@custom_jit
def optimize_Adam_wpm(model_fn, params, step, batch, grads, state, lambd, weight_decay):
    return damp_update(params, step, batch, grads, state, lambd, weight_decay, model_fn, get_optim_wpm)

@jax.jit
def optimize_Adam(params, step, grads, state):
    return adam_update(params, step, grads, state)

