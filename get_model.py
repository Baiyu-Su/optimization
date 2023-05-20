import flaxmodels as fm

def get_model():
    return fm.ResNet34(output='logits', pretrained=None, num_classes=10)
    
