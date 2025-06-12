# Iris Classifier with Neural Network

This project uses a neural network built with TensorFlow/Keras to classify iris flowers based on the Iris dataset from the UCI Machine Learning Repository.

# Dataset

- **Source:** [UCI ML Repository - Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- **Features:** 4 (sepal length, sepal width, petal length, petal width)
- **Classes:** Setosa, Versicolor, Virginica

# Model Architecture

- Input: 4 features
- Hidden layers:
  - Dense(32), ReLU
  - Dense(8), ReLU
- Output: Dense(3) (used with `SparseCategoricalCrossentropy` and `from_logits=True`)

# Results

- Accuracy: ~96–97% (may vary slightly due to randomness)
