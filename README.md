# S7-Assignment-Solution

This repository contains the necessary files and notebooks for training a CIFAR-10 image classification model. The model architecture is  described as follows:

```css
C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP c10
```

Where:

- `C1`, `C2`, `C3`, `C4`, `C5`, `C7`, `C8`, and `C9` represent convolutional layers.
- `c3`, `c6`, and `C10` are 1x1 convolutional layers.
- `P1` and `P2` represent max pooling layers.
- `GAP` denotes the Global Average Pooling layer.

The total number of parameters for the model is 48,178, which is below the specified limit of 50,000. Additionally, there are two skip connections present in the second and third blocks.



The repository includes the following files:

### Notebooks

1. `ERA-S8-BN.ipynb`: A Jupyter notebook that demonstrates training the CIFAR-10 model using batch normalization.
2. `ERA-S8-GN.ipynb`: A Jupyter notebook that demonstrates training the CIFAR-10 model using group normalization.
3. `ERA-S8-LN.ipynb`: A Jupyter notebook that demonstrates training the CIFAR-10 model using layer normalization.

### Python Scripts

1. `backprop.py`: Contains the implementation of the backpropagation algorithm used for training the model.
2. `dataset.py`: Provides functions to load and preprocess the CIFAR-10 dataset.
3. `model.py`: Defines the architecture of the CIFAR-10 model.
4. `scheduler.py`: Implements learning rate scheduling techniques for optimizing the training process.
5. `training.py`: Contains functions and utilities for training the CIFAR-10 model.
6. `transform.py`: Defines various image transformations and augmentation techniques.
7. `utils.py`: Includes utility functions used throughout the training process.
8. `visualize.py`: Provides functions for visualizing the training progress and model performance.





### Accuracy

#### 1. Batch Normalization

- Training Accuracy: 78.42%
- Test Accuracy: 80.08%

#### 2. Group Normalization

- Training Accuracy: 73.19%
- Test Accuracy: 72.38%

#### 3. Layer Normalization

- Training Accuracy: 73.22%
- Test Accuracy: 72.38%