# Run MNIST example
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pycatml.categories.image import ImageCategory
from pycatml.functors.base import Functor

def main():
    # --- Define the image category
    image_category = ImageCategory()

    # --- Load MNIST dataset
    transform_pipeline = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)
    dataloader = DataLoader(mnist, batch_size=1, shuffle=True)

    # --- Take a single sample
    img_tensor, label = next(iter(dataloader))

    print(f"Raw MNIST Sample - Label: {label.item()} Shape: {img_tensor.shape}")

    # --- Define image transformations as morphisms
    # Each transformation takes a tensor and outputs a transformed tensor
    resize_fn = transforms.Resize((32, 32))
    grayscale_fn = transforms.Grayscale()
    normalize_fn = transforms.Normalize((0.5,), (0.5,))

    img_category = ImageCategory()
    img_category.add_image_object(img_tensor, {"label": label.item(), "shape": img_tensor.shape})

    # Apply morphism (resize)
    resized = resize_fn(img_tensor)
    img_category.add_image_object(resized, {"shape": resized.shape})
    img_category.add_image_morphism(img_tensor, resized, resize_fn)

    # --- Define functor: pipeline = Normalize ∘ Resize
    pipeline_functor = Functor(
        obj_map=lambda x: normalize_fn(resize_fn(x)),
        morph_map=lambda f: lambda x: normalize_fn(resize_fn(f(x)))
    )

    transformed_image = pipeline_functor.apply_obj(img_tensor)
    print(f"Transformed image shape: {transformed_image.shape}")

if __name__ == "__main__":
    main()
