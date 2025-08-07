# PyCatML

Composable Machine Learning Pipelines via Category-Theoretic Abstractions

PyCatML is a Python framework implementing the theoretical foundations proposed in the paper:  
**"Composable Machine Learning Pipelines via Category-Theoretic Abstractions: A Functorial Approach to Modular Data Processing"**

It provides a mathematical and computational infrastructure for modular, interpretable, and reusable machine learning pipelines using Category Theory.

## Features

- Category Abstractions (`Category`, `ImageCategory`, `TextCategory`, etc.)
- Functor Classes representing end-to-end pipelines
- Natural Transformations for adaptive and interpretable computation
- Multimodal Fusion Pipelines using product categories
- Gradient Category and Optimizers with pullbacks and adjoint transformations
- Pipeline Composition, Data Augmentation, and Explanation

## Project Structure

```
pycatml/
├── categories/
│   ├── base.py
│   ├── image.py
│   ├── text.py
│   ├── multimodal.py
│   └── rl.py
├── functors/
│   ├── base.py
│   ├── augmentation.py
│   └── fusion.py
├── transformations/
│   ├── natural_transformation.py
│   └── explanation.py
├── pipelines/
│   ├── mnist_pipeline.py
│   └── multimodal_pipeline.py
├── optimizer/
│   ├── gradient_category.py
│   └── categorical_optimizer.py
├── examples/
│   ├── run_mnist.py
│   └── run_multimodal.py
├── tests/
│   └── test_pipeline.py
├── utils/
│   └── validation.py
├── setup.py
└── README.md
```

## Getting Started

```bash
git clone https://github.com/pouyashaeri/pycatml.git
cd pycatml
pip install -e .
```

## Examples

### MNIST Preprocessing Functor

```python
from pycatml.pipelines.mnist_pipeline import MNISTPipeline

pipeline = MNISTPipeline()
image, label = pipeline.run()
```

### Multimodal Fusion

```python
from pycatml.pipelines.multimodal_pipeline import MultimodalPipeline

pipeline = MultimodalPipeline(vision_encoder, text_encoder, fusion_module)
fused_output = pipeline.run()
```

## Evaluation

We provide benchmarks for:
- Accuracy
- Compositional Overhead
- Adaptivity Effectiveness
- Interpretability Gains

See `examples/` and `tests/` for runnable demos.

