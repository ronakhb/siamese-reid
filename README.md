# Person Re-Identification Baseline System

This repository contains a baseline system for person re-identification (ReID). It includes training and evaluation scripts, as well as utility functions for data loading, model building, loss computation, and optimization.

## Files

### `main.py`

This script serves as the entry point for training and testing the person ReID model. It accepts various command-line arguments to configure the training procedure, such as batch size, learning rate, model architecture, and optimization parameters.

### `data.py`

This module contains functions to create data loaders for the person ReID dataset. It provides methods to preprocess the input images and organize them into batches for training and testing.

### `engine/trainer.py`

The `trainer.py` module defines the training loop for the person ReID model. It includes functions to perform forward and backward passes, compute loss values, update model parameters, and track performance metrics during training.

### `modeling.py`

This file defines the architecture of the person ReID model. It includes functions to build different variants of the model, such as siamese networks or baseline architectures, with configurable parameters.

### `loss.py`

The `loss.py` module contains functions to compute loss values for the person ReID task. It implements various loss functions, including center loss and range loss, to optimize feature embeddings for improved person re-identification performance.

### `solver.py`

This file provides utility functions to create optimizers and learning rate schedulers for training the person ReID model. It includes implementations of popular optimization algorithms such as SGD and Adam, as well as custom learning rate schedulers for efficient training.

### `engine/inference.py`

The `inference.py` module contains functions to evaluate the trained model on unseen data. It includes methods to perform inference on test images, compute similarity scores between query and gallery images, and generate evaluation metrics such as mean Average Precision (mAP) and Cumulative Matching Characteristics (CMC) curves.

### `onnxInference.py`

This file contains functions to perform inference using ONNX (Open Neural Network Exchange) format. It provides methods to load ONNX models, execute inference, and analyze performance metrics.

## Usage

To train the person ReID model, run the `main.py` script with appropriate command-line arguments specifying the training configuration. For example:

```bash
python main.py --batch_size 64 --lr 0.00035 --model_name siamese --max_epochs 50 --train 1
