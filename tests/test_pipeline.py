# Test for categorical pipelines
from ..pipelines.mnist_pipeline import MNISTPipeline

def test_mnist_pipeline():
    pipeline = MNISTPipeline()
    processed_img, label = pipeline.run()

    assert processed_img.shape[-2:] == (28, 28), "Image not properly resized"
    assert -1.0 <= processed_img.min() <= processed_img.max() <= 1.0, "Normalization failed"
    print("MNIST pipeline test passed")

if __name__ == "__main__":
    test_mnist_pipeline()
