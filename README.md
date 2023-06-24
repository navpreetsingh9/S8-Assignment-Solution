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

   

## Outcome

### 1. Batch Normalization model

#### Accuracy

- Training Accuracy: 78.42%
- Test Accuracy: 80.08%

![BN_Result](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/bn_result.png)

#### Findings

- Batch Size: Increasing the batch size did not significantly impact the performance of the batch normalization model. Various batch sizes were tested, and all resulted in satisfactory accuracy.
- Learning Rate: The learning rate used for training the batch normalization model provided good results without the need for significant adjustments.
- Performance: The batch normalization model consistently outperformed the group normalization and layer normalization models, achieving higher accuracy on the CIFAR-10 dataset.

#### Incorrect Classified Images

![BN_Incorrect](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/bn_incorrect.png)

### 2. Group Normalization model (Group Size = 2)

#### Accuracy

- Training Accuracy: 73.19%
- Test Accuracy: 72.38%

![GN_Result](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/gn_result.png)

#### Findings

- Batch Size: Increasing the batch size had a negative impact on the performance of the group normalization model. A batch size of 32 resulted in the best accuracy (>70%).
- Learning Rate: The group normalization model required a lower learning rate compared to the batch normalization model to achieve better accuracy.
- Performance: Despite using an optimal batch size and adjusting the learning rate, the group normalization model did not perform as well as the batch normalization model, but comparable to the accuracy of layer normalization model on the CIFAR-10 dataset.

#### Incorrect Classified Images

![GN_Incorrect](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/gn_incorrect.png)

### 3. Layer Normalization model

#### Accuracy

- Training Accuracy: 73.22%
- Test Accuracy: 72.38%

![LN_Result](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/ln_result.png)

#### Findings

- Batch Size: Similar to the group normalization model, increasing the batch size had a negative impact on the performance of the layer normalization model. A batch size of 32 was found to be optimal for achieving more than 70% accuracy.
- Learning Rate: The layer normalization model also required a lower learning rate compared to the batch normalization model for achieving better accuracy.
- Performance: Like the group normalization model, the layer normalization model fell short in performance compared to the batch normalization model, but comparable to the accuracy of group normalization model on the CIFAR-10 dataset.

#### Incorrect Classified Images

![LN_Incorrect](https://github.com/navpreetsingh9/S8-Assignment-Solution/blob/ce3a9200308e7f36a015155e679f8db554db853a/img/ln_incorrect.png)

