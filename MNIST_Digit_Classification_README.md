
# MNIST Digit Classification Model

This project demonstrates the steps involved in building, training, and evaluating a machine learning model for digit classification using the MNIST dataset. The model is built using TensorFlow and Keras, and evaluates performance using various metrics like accuracy and loss.

## Steps involved:

### 1. Import Libraries
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
```

### 2. Load and Preprocess the Data
The MNIST dataset is loaded, and the training and test data are normalized.

```python
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0
```

### 3. Visualizing an Image from the Dataset
We can visualize the first image from the training data using `plt.imshow`.

```python
plt.imshow(X_train[0], cmap='gray')
plt.show()
```

### 4. Build the Model
A simple neural network model is built using the Keras Sequential API.

```python
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the input image
model.add(Dense(128, activation='relu'))  # First hidden layer
model.add(Dense(32, activation='relu'))   # Second hidden layer
model.add(Dense(10, activation='softmax'))  # Output layer for 10 classes
```

### 5. Compile the Model
The model is compiled with the Adam optimizer and sparse categorical cross-entropy loss function.

```python
model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
```

### 6. Train the Model
The model is trained using the training data with a validation split of 0.2.

```python
history = model.fit(X_train, y_train, epochs=25, validation_split=0.2)
```

### 7. Visualize Training & Validation Metrics
Plotting the accuracy and loss for both training and validation sets.

```python
# Plotting accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

# Plotting loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

### 8. Evaluate the Model on Test Data
The model is tested on unseen test data, and accuracy is calculated.

```python
from sklearn.metrics import accuracy_score

y_prob = model.predict(X_test)
y_pred = y_prob.argmax(axis=1)  # Convert probabilities to predicted class labels

accuracy_score(y_test, y_pred)  # Calculate the accuracy of the model
```

### 9. Visualizing Predictions
We can visualize individual test images and the corresponding predictions:

```python
plt.imshow(X_test[1], cmap='gray')
plt.show()

# Predict for the second test image
model.predict(X_test[1].reshape(1, 28, 28)).argmax(axis=1)
```

## Conclusion
This notebook covers all the essential steps to build and evaluate a machine learning model for classifying handwritten digits using the MNIST dataset. It includes data preprocessing, model building, training, evaluation, and visualization.
