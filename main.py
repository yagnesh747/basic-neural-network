import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
import urllib.request

BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
DATA_DIR = "data"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz"
}


def download_file(filename):
    os.makedirs(DATA_DIR, exist_ok=True)

    path = os.path.join(DATA_DIR, filename)

    if not os.path.exists(path):
        url = BASE_URL + filename
        print("Downloading:", filename)
        urllib.request.urlretrieve(url, path)

    return path


def load_images(filename):
    path = download_file(filename)

    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)

    return data.reshape(-1, 784).astype(np.float32) / 255.0


def load_labels(filename):
    path = download_file(filename)

    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

    return data.astype(np.int64)


def one_hot(labels, classes=10):
    result = np.zeros((labels.size, classes))

    result[np.arange(labels.size), labels] = 1

    return result


class NeuralNetwork:

    def __init__(self, input_size=784, hidden_size=128, output_size=10):

        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))


    def relu(self, x):
        return np.maximum(0, x)


    def relu_derivative(self, x):
        return (x > 0).astype(float)


    def softmax(self, x):
        x = x - np.max(x, axis=1, keepdims=True)

        exp_x = np.exp(x)

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)


    def forward(self, X):

        self.Z1 = np.dot(X, self.W1) + self.b1

        self.A1 = self.relu(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2

        self.A2 = self.softmax(self.Z2)

        return self.A2


    def loss(self, Y, predictions):

        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)

        return -np.mean(
            np.sum(Y * np.log(predictions), axis=1)
        )


    def backward(self, X, Y, learning_rate):

        m = X.shape[0]

        dZ2 = self.A2 - Y

        dW2 = np.dot(self.A1.T, dZ2) / m

        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)

        dZ1 = dA1 * self.relu_derivative(self.Z1)

        dW1 = np.dot(X.T, dZ1) / m

        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1


    def predict(self, X):

        probabilities = self.forward(X)

        return np.argmax(probabilities, axis=1)


def accuracy(y_true, y_pred):

    return np.mean(y_true == y_pred) * 100


def main():

    np.random.seed(42)

    print("Loading MNIST dataset...")

    X_train = load_images(FILES["train_images"])
    y_train = load_labels(FILES["train_labels"])

    X_test = load_images(FILES["test_images"])
    y_test = load_labels(FILES["test_labels"])

    print("Training samples:", X_train.shape)
    print("Testing samples:", X_test.shape)

    # Use a smaller subset for faster laptop training
    X_train = X_train[:20000]
    y_train = y_train[:20000]

    X_test = X_test[:3000]
    y_test = y_test[:3000]

    Y_train = one_hot(y_train)

    model = NeuralNetwork()

    epochs = 15
    batch_size = 64
    learning_rate = 0.01

    losses = []
    accuracies = []

    for epoch in range(epochs):

        indices = np.random.permutation(len(X_train))

        X_train = X_train[indices]
        Y_train = Y_train[indices]
        y_train = y_train[indices]

        epoch_loss = 0

        for start in range(0, len(X_train), batch_size):

            end = start + batch_size

            X_batch = X_train[start:end]
            Y_batch = Y_train[start:end]

            predictions = model.forward(X_batch)

            batch_loss = model.loss(Y_batch, predictions)

            model.backward(
                X_batch,
                Y_batch,
                learning_rate
            )

            epoch_loss += batch_loss

        epoch_loss /= (len(X_train) // batch_size)

        train_predictions = model.predict(X_train)

        train_accuracy = accuracy(
            y_train,
            train_predictions
        )

        losses.append(epoch_loss)
        accuracies.append(train_accuracy)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Loss: {epoch_loss:.4f} "
            f"- Accuracy: {train_accuracy:.2f}%"
        )

    test_predictions = model.predict(X_test)

    test_accuracy = accuracy(
        y_test,
        test_predictions
    )

    print("\nFinal Test Accuracy:", f"{test_accuracy:.2f}%")

    os.makedirs("results", exist_ok=True)

    # Loss graph
    plt.figure(figsize=(8, 5))

    plt.plot(losses)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training Loss")

    plt.grid()

    plt.savefig("results/loss_curve.png")

    plt.show()

    # Prediction visualization
    plt.figure(figsize=(10, 5))

    for i in range(10):

        plt.subplot(2, 5, i + 1)

        plt.imshow(
            X_test[i].reshape(28, 28),
            cmap="gray"
        )

        plt.title(
            f"Actual: {y_test[i]}\n"
            f"Predicted: {test_predictions[i]}"
        )

        plt.axis("off")

    plt.tight_layout()

    plt.savefig("results/predictions.png")

    plt.show()


if __name__ == "__main__":
    main()