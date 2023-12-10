#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = tf.one_hot(y_train, 10)
y_test_one_hot = tf.one_hot(y_test, 10)

# Define the neural network architecture using Keras
def build_simple_model(NN_width):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(NN_width, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer with 10 classes and softmax activation
    ])
    return model

# Define the parameters
NN_width = 128
learning_rate = 0.001

# Build the model
simple_model = build_simple_model(NN_width)

# Compile the model
simple_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Training the model without cross-validation
history = simple_model.fit(X_train, y_train_one_hot, epochs=10)

# Plot training loss and accuracy
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='green')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy = simple_model.evaluate(X_test, y_test_one_hot)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[11]:


# Assuming you've trained the model as described in the previous code
# ...

# Make predictions on the test set
predictions = simple_model.predict(X_test)

# Convert predictions to class labels
predicted_labels = np.argmax(predictions, axis=1)

# Display some predictions
for i in range(10):
    print(f"True Label: {y_test[i]}, Predicted Label: {predicted_labels[i]}")

# Evaluate the model on the test set
test_loss, test_accuracy = simple_model.evaluate(X_test, y_test_one_hot)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")


# In[ ]:




