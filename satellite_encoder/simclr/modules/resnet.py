import torchvision


def get_resnet(name, pretrained=False):
    resnet = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnet.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnet[name]
