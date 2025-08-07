# MNIST categorical pipeline
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from ..categories.image import ImageCategory
from ..functors.base import Functor

class MNISTPipeline:
    def __init__(self):
        self.image_category = ImageCategory()
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.pipeline_functor = Functor(
            obj_map=self.transform,
            morph_map=lambda f: lambda x: self.transform(f(x))
        )

    def load_sample(self):
        dataset = MNIST(root='./data', train=True, download=True, transform=None)
        img, label = dataset[0]
        tensor_img = transforms.ToTensor()(img)
        self.image_category.add_image_object(tensor_img, {"label": label, "source": "raw"})
        return tensor_img, label

    def run(self):
        raw_img, label = self.load_sample()
        processed_img = self.pipeline_functor.apply_obj(raw_img)
        self.image_category.add_image_object(processed_img, {"label": label, "stage": "processed"})
        return processed_img, label
