"""
A stochastic data augmentation module that transforms any given data example randomly
resulting in two correlated views of the same example,
denoted x ̃i and x ̃j, which we consider as a positive pair.
"""

import torchvision


class TransformsSimCLR:
    def __init__(self, size):

        self.train_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.RandomResizedCrop(size=size),
                # torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                # torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        return self.train_transform(x)
