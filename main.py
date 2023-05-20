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
import numpy as np

from loss_utils import loss
from func_utils import dot_product
from data_utils import to_numpy_iterator, create_dataloader
from optimizer import adam_init, optimize_Adam, optimize_AdamK
from get_model import get_model


def train_Adam(initial_params, train_ds, test_ds, batch_size, learning_rate, seed=None):
    params = initial_params

    trainloader, testloader = create_dataloader(train_ds, test_ds, batch_size, seed)

    # Convert the PyTorch DataLoaders to NumPy iterators
    train_ds_numpy = to_numpy_iterator(trainloader)
    test_ds_numpy = to_numpy_iterator(testloader)

    # Create a function to compute gradients
    fixed_loss = Partial(loss, model)
    loss_and_grads = jax.value_and_grad(fixed_loss, argnums=0, has_aux=True)

    # Initialize optimizer state
    state = adam_init(params, learning_rate=learning_rate)

    # Training loop
    num_steps = 1000
    train_loss_list = []
    test_loss_list = []

    start_time = time.time()
    elapsed_time = 0

    for step in tqdm(range(num_steps)):
        batch = next(train_ds_numpy)
        test_batch = next(test_ds_numpy)

        (loss_value, logits), gradients = loss_and_grads(params, batch)
        params, state = optimize_Adam(params, step, gradients, state)

        accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])
        test_loss_value, test_logits = fixed_loss(params, test_batch)
        test_accuracy = jnp.mean(jnp.argmax(test_logits, -1) == test_batch[1])
        train_loss_list.append(loss_value)
        test_loss_list.append(test_loss_value)

        elapsed_time = time.time() - start_time
        print(f'''Step {step}, train loss {loss_value:.3f}, test loss {test_loss_value:.3f}, 
        train accuracy {accuracy:.3f}, test accuracy {test_accuracy:.3f}, elapsed time {elapsed_time:.2f}.''')
        
    return params, train_loss_list, test_loss_list

if __name__ == '__main__':
    model = get_model()

    # Model parameters for AdamK and Adam
    batch_size = 64
    # Define a transformation to apply to the images
    transform = transforms.Compose(
     [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    key1, key2 = random.split(random.PRNGKey(42))
    params1 = model.init(key2, jnp.ones((batch_size, 32, 32, 3)))
    params2 = jnp.copy(params1)
    params3 = jnp.copy(params1)
    params4 = jnp.copy(params1)
    params5 = jnp.copy(params1)

    trainset1 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset1 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    trained_params1, train_loss1, test_loss1 = train_Adam(params1, trainset1, testset1, batch_size, 0.1, seed=42)

    trainset2 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset2 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    trained_params2, train_loss2, test_loss2 = train_Adam(params2, trainset2, testset2, batch_size, 0.2, seed=42)

    trainset3 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset3 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    trained_params3, train_loss3, test_loss3 = train_Adam(params3, trainset3, testset3, batch_size, 0.5, seed=42)

    trainset4 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset4 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    trained_params4, train_loss4, test_loss4 = train_Adam(params4, trainset4, testset4, batch_size, 1.0, seed=42)

    trainset5 = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset5 = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    trained_params5, train_loss5, test_loss5 = train_Adam(params5, trainset5, testset5, batch_size, 2.0, seed=42)

    train_arr1 = np.array(train_loss1)
    train_arr2 = np.array(train_loss2)
    train_arr3 = np.array(train_loss3)
    train_arr4 = np.array(train_loss4)
    train_arr5 = np.array(train_loss5)

    test_arr1 = np.array(test_loss1)
    test_arr2 = np.array(test_loss2)
    test_arr3 = np.array(test_loss3)
    test_arr4 = np.array(test_loss4)
    test_arr5 = np.array(test_loss5)

    np.savez('loss.npz', train_arr1, train_arr2, train_arr3, train_arr4, train_arr5,
             test_arr1, test_arr2, test_arr3, test_arr4, test_arr5)

