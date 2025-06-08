import os
import numpy as np
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_flat = X_train.reshape(-1, 784)
X_test_flat = X_test.reshape(-1, 784)


train_data = np.column_stack((y_train, X_train_flat))
test_data = np.column_stack((y_test, X_test_flat))

os.makedirs("MNISTdata", exist_ok=True)
np.savetxt("MNISTdata/mnist_train.csv", train_data, delimiter=",", fmt="%d")
np.savetxt("MNISTdata/mnist_test.csv", test_data, delimiter=",", fmt="%d")

print("MNIST CSVs saved to MNISTdata/")
