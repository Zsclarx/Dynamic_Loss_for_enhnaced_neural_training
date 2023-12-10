#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tensorflow as tf
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert labels to one-hot encoding
y_train_one_hot = tf.one_hot(y_train, 10)
y_test_one_hot = tf.one_hot(y_test, 10)

# Define the neural network architecture using Keras
def build_model(NN_width):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(NN_width, activation='relu'),
        tf.keras.layers.Dense(10)
    ])
    return model

# Weights for the loss function
def c_fn(t, i, w_max):
    slope = 2 * (w_max - 1) / T
    w_main_class = tf.where(t < T / 2., 1 + t * slope, 2 * w_max - t * slope - 1)
    res = tf.ones(C) + (w_main_class - 1) * tf.one_hot(i, C, dtype=tf.float32)
    res = res / tf.reduce_sum(res) * C
    return res

# Dynamical loss function
def weighted_loss(model, X, Y, t, i, w_max):
    w = c_fn(t, i, w_max)
    predictions = model(X)

    # Cross-entropy loss
    cross_entropy_loss = -tf.reduce_mean(tf.nn.log_softmax(predictions) * Y * w) * C

    return cross_entropy_loss

# Standard loss function
def loss(model, X, Y):
    predictions = model(X)
    cross_entropy_loss = -tf.reduce_mean(tf.nn.log_softmax(predictions) * Y) * C
    return cross_entropy_loss

# Accuracy
def accuracy(model, X, Y):
    predictions = model(X)
    acc = tf.reduce_mean(tf.cast(tf.argmax(predictions, axis=1) == tf.argmax(Y, axis=1), tf.float32))
    return acc

# Function that takes one minimization step
def step_CE(optimizer, model, X, Y, t, i, w_max):
    with tf.GradientTape() as tape:
        loss_value = weighted_loss(model, X, Y, t, i, w_max)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_value

# Define the parameters
NN_width = 512
learning_rate = 0.001
T = 10  # You need to set the value for T
C = 10  # For MNIST, C is 10 as there are 10 classes

# Build the model
model = build_model(NN_width)

# Create an optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Training loop
total_time = 1500
w_max = 1.5  # You need to set the initial value for w_max

for i in range(total_time):
    t = i % T
    c = i % C
    w_max = max(1, w_max - 0.001)  # Adjust this according to your needs

    # Reshape input data for the Flatten layer
    X_train_reshaped = X_train.reshape((X_train.shape[0], 28, 28))

    # Convert to TensorFlow tensors
    X_train_tf = tf.convert_to_tensor(X_train_reshaped, dtype=tf.float32)
    Y_train_tf = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)

    loss_value = step_CE(optimizer, model, X_train_tf, Y_train_tf, t, c, w_max)

    if i % 100 == 0:
        acc = accuracy(model, X_train_tf, Y_train_tf)
        print(f"Step {i}, Loss: {loss_value.numpy()}, Accuracy: {acc.numpy() * 100}%")

# After training, you can use the model for predictions, e.g., model.predict(X_test_tf)


# In[8]:


X_test_reshaped = X_test.reshape((X_test.shape[0], 28, 28))
X_test_tf = tf.convert_to_tensor(X_test_reshaped, dtype=tf.float32)

# Make predictions
predictions = model(X_test_tf)

# Convert predictions to class labels
predicted_labels = tf.argmax(predictions, axis=1).numpy()

# Display predicted labels
print("Predicted Labels:", predicted_labels)

test_accuracy = np.mean(predicted_labels == y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")



# In[9]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))

for i, ax in enumerate(axes):
    ax.imshow(X_test[i], cmap='gray')
    ax.set_title(f"True: {y_test[i]}\nPred: {predicted_labels[i]}", fontsize=8)
    ax.axis('off')

plt.show()


# In[13]:


training_loss_history = [44.61333453655243,95.42666673660278,97.7400004863739,98.8183319568634,99.41999912261963,99.75666403770447,99.90333318710327,99.96333122253418,99.98833537101746,99.99499917030334,99.99833106994629,100.0,100.0,100.0,100.0]


# In[14]:


# Plot training accuracy
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(training_loss_history, label='Training Loss', color='blue')
plt.title('Training Loss')
plt.xlabel('Training Step')
plt.ylabel('Loss')
plt.xticks(np.arange(0, len(training_loss_history), step=100), labels=np.arange(0, len(training_loss_history)//100))
plt.legend()


# In[ ]:




