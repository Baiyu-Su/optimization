import numpy as np
import torch


def to_numpy_iterator(dataloader):
    while True:
        for images, labels in iter(dataloader):
            # Transpose the images to have the channel dimension at the end (NCHW -> NHWC)
            yield np.transpose(images.numpy(), (0, 2, 3, 1)), labels.numpy()

def create_dataloader(trainset, testset, batch_size, seed=None):
    if seed is not None:
        generator = torch.Generator().manual_seed(seed)
    else:
        generator = None

    # Create PyTorch DataLoader objects for the train and test sets
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2, generator=generator)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2, generator=generator)

    return trainloader, testloader