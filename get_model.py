from jax_resnet import ResNet50

def get_model(num_classes):
    return ResNet50(n_classes=num_classes)
    
