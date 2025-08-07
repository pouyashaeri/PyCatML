# Data Augmentation Endofunctor
from torchvision import transforms
from .base import Functor

def get_augmentation_functor():
    augmentation_pipeline = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2)
    ])

    return Functor(
        obj_map=lambda img: augmentation_pipeline(img),
        morph_map=lambda f: lambda x: augmentation_pipeline(f(x))
    )
