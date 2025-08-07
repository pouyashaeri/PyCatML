from setuptools import setup, find_packages

setup(
    name="pycatml",
    version="0.1.0",
    description="Composable Machine Learning Pipelines via Category-Theoretic Abstractions",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Pouya Shaeri and Arash Karimi",
    url="https://github.com/pouyashaeri/PyCatML",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy",
        "scikit-learn"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
