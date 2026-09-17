# Basic Neural Network from Scratch

A simple neural network implemented from scratch using **Python and NumPy** to classify handwritten digits from the MNIST dataset.

This project was developed as part of the **AI Intern – Week 1: Foundations of Artificial Intelligence** task.

---

## 📌 Project Overview

The objective of this project is to understand the fundamental working of a neural network by implementing the complete training process without using machine learning frameworks such as TensorFlow, PyTorch, or Scikit-learn.

The neural network takes handwritten digit images as input and predicts one of the ten classes from **0 to 9**.

---

## 🎯 Objectives

- Understand the basic architecture of a neural network
- Implement forward propagation
- Implement activation functions
- Calculate cross-entropy loss
- Implement backpropagation
- Update network weights using gradient descent
- Calculate classification accuracy
- Visualize the training loss
- Predict handwritten digits

---

## 🧠 Neural Network Architecture

The model consists of three main layers:

```text
Input Layer
784 neurons
   ↓
Hidden Layer
128 neurons
ReLU Activation
   ↓
Output Layer
10 neurons
Softmax Activation
   ↓
Digit Prediction
0 - 9