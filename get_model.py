from jax_resnet import ResNet18

def get_model(num_classes):
    return ResNet18(n_classes=num_classes)
    
