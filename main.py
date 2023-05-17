# Import vision dataset from PyTorch and convert data to numpy compatible form
import os

os.environ['CPATH'] = os.path.expanduser('~/cudnn/usr/include') + ':' + os.environ.get('CPATH', '')
os.environ['LIBRARY_PATH'] = os.path.expanduser('~/cudnn/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LIBRARY_PATH', '')
os.environ['LD_LIBRARY_PATH'] = os.path.expanduser('~/cudnn/usr/lib/x86_64-linux-gnu') + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import torchvision
import torchvision.transforms as transforms

import time
from tqdm import tqdm
from jax.tree_util import Partial

import jax
import jax.numpy as jnp
from jax import random, grad

from loss_utils import loss
from func_utils import dot_product
from data_utils import to_numpy_iterator, create_dataloader
from optimizer import adam_init, optimize_Adam, optimize_AdamK
from get_model import get_model


def train_AdamK(initial_params, train_ds, test_ds, seed=None):
    params = initial_params

    trainloader, testloader = create_dataloader(train_ds, test_ds, 64, seed)

    # Convert the PyTorch DataLoaders to NumPy iterators
    train_ds_numpy = to_numpy_iterator(trainloader)
    test_ds_numpy = to_numpy_iterator(testloader)
    
    # Training constant for AdamK
    T1 = 5
    omega1 = (9 / 10) ** T1
    lambd = 3
    weight_decay = 1e-4
    
    def model_fn(params, inputs):
        return model.apply(params, inputs)
    
    # Create a function to compute gradients
    fixed_loss = Partial(loss, model)
    loss_and_grads = jax.value_and_grad(fixed_loss, argnums=0, has_aux=True)

    # Initialize optimizer state
    state = adam_init(params, learning_rate=1.5)

    # Training loop
    num_steps = 400
    train_loss_list = []
    valid_loss_list = []
    valid_loss_time = []

    start_time = time.time()
    elapsed_time = 0
    for j in tqdm(range(num_steps)):
        batch = next(train_ds_numpy)

        (loss_value, logits), gradients = loss_and_grads(params, batch)
        params, state = optimize_AdamK(model_fn, params, batch, gradients, state, lambd, weight_decay)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])

        if j % T1 == 0:
            next_loss = fixed_loss(params, batch)
            rho = -(next_loss - loss_value) / (0.5 * dot_product(gradients, state['damps']))
            print(rho)
            if rho > 3/4:
                lambd = omega1 * lambd
            if rho < 1/4:
                lambd = lambd / omega1

        train_loss_list.append(loss_value)

        elapsed_time = time.time() - start_time

        print(f"This is step {j}, train loss: {loss_value:.3f}, train accuracy: {accuracy:.3f}, using lambda {lambd:.6f}, elapsed time: {elapsed_time:.2f} seconds.")

    return params, train_loss_list, valid_loss_list, valid_loss_time


def train_Adam(initial_params, train_ds, test_ds, seed=None):
    params = initial_params

    trainloader, testloader = create_dataloader(train_ds, test_ds, 64, seed)

    # Convert the PyTorch DataLoaders to NumPy iterators
    train_ds_numpy = to_numpy_iterator(trainloader)
    test_ds_numpy = to_numpy_iterator(testloader)

    # Create a function to compute gradients
    fixed_loss = Partial(loss, model)
    loss_and_grads = jax.value_and_grad(fixed_loss, argnums=0, has_aux=True)

    # Initialize optimizer state
    state = adam_init(params, learning_rate=1)

    # Training loop
    num_steps = 600
    train_loss_list = []
    valid_loss_list = []
    valid_loss_time = []

    start_time = time.time()
    elapsed_time = 0

    for j in tqdm(range(num_steps)):
        batch = next(train_ds_numpy)

        (loss_value, logits), gradients = loss_and_grads(params, batch)
        params, state = optimize_Adam(params, gradients, state)
        train_loss_list.append(loss_value)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])

        elapsed_time = time.time() - start_time
        print(f"This is step {j}, train loss: {loss_value:.3f}, train accuracy: {accuracy:.3f}, elapsed time: {elapsed_time:.2f} seconds.")

    return params, train_loss_list, valid_loss_list, valid_loss_time

if __name__ == '__main__':
    model = get_model(num_classes=10)

    # Model parameters for AdamK and Adam
    batch_size = 64
    key1, key2 = random.split(random.PRNGKey(42))
    params = model.init(key2, jnp.ones((batch_size, 32, 32, 3)))
    params2 = model.init(key2, jnp.ones((batch_size, 32, 32, 3)))

    # Define a transformation to apply to the images
    transform = transforms.Compose(
     [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)
    
    seed = 42
    trained_params1, train_loss1, valid_loss1, valid_loss_time1 = train_AdamK(params, trainset, testset, seed=seed)

    # Load the CIFAR-10 dataset
    trainset2 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=transform)
    testset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=False, transform=transform)

    trained_params2, train_loss2, valid_loss2, valid_loss_time2 = train_Adam(params2, trainset2, testset2, seed=seed)
