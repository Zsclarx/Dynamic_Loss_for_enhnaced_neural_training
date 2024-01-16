Name – Prakhar Pratap Singh 

Roll No. – 22075102 and 22074024 

Group No. – 63 (Prakhar Pratap Singh and Pratham Agarwal) 

**Google Colab link :-**  

[https://colab.research.google.com/drive/1fNNzxhwEc4jdvBdQa-Xj_DnUs3ENxrKP?usp=sharing](https://colab.research.google.com/drive/1fNNzxhwEc4jdvBdQa-Xj_DnUs3ENxrKP?usp=sharing)

The DynamicLoss function is designed to be a flexible and customizable loss function in TensorFlow/Keras. Let's break down each part of the code.
### Dynamical Loss Function Implementation Documentation:

#### 1. **Load and Preprocess Data:**
   - The code loads the MNIST dataset and preprocesses the images and labels.
     ```python
     mnist = tf.keras.datasets.mnist
     (X_train, y_train), (X_test, y_test) = mnist.load_data()
     X_train, X_test = X_train / 255.0, X_test / 255.0
     y_train_one_hot = tf.one_hot(y_train, 10)
     y_test_one_hot = tf.one_hot(y_test, 10)
     ```

#### 2. **Neural Network Architecture:**
   - The neural network architecture is defined using Keras. It's a simple feedforward network.
     ```python
     def build_model(NN_width):
         model = tf.keras.Sequential([
             tf.keras.layers.Flatten(input_shape=(28, 28)),
             tf.keras.layers.Dense(NN_width, activation='relu'),
             tf.keras.layers.Dense(10)
         ])
         return model
     ```

#### 3. **Dynamical Loss Function (Weighted Loss):**
   - A custom dynamical loss function `weighted_loss` is defined. The weights for the loss function are determined by the function `c_fn`.
     ```python
     def c_fn(t, i, w_max):
         # ... (calculates weights based on time 't', class index 'i', and maximum weight 'w_max')
         return res

     def weighted_loss(model, X, Y, t, i, w_max):
         w = c_fn(t, i, w_max)
         predictions = model(X)
         cross_entropy_loss = -tf.reduce_mean(tf.nn.log_softmax(predictions) * Y * w) * C
         return cross_entropy_loss
     ```

#### 4. **Standard Loss Function:**
   - A standard cross-entropy loss function is defined for comparison during training.
     ```python
     def loss(model, X, Y):
         predictions = model(X)
         cross_entropy_loss = -tf.reduce_mean(tf.nn.log_softmax(predictions) * Y) * C
         return cross_entropy_loss
     ```

#### 5. **Accuracy Calculation:**
   - A function to calculate accuracy is defined.
     ```python
     def accuracy(model, X, Y):
         predictions = model(X)
         acc = tf.reduce_mean(tf.cast(tf.argmax(predictions, axis=1) == tf.argmax(Y, axis=1), tf.float32))
         return acc
     ```

#### 6. **Training Loop:**
   - The training loop uses the custom loss function and updates the model's weights accordingly.
     ```python
     def step_CE(optimizer, model, X, Y, t, i, w_max):
         with tf.GradientTape() as tape:
             loss_value = weighted_loss(model, X, Y, t, i, w_max)
         gradients = tape.gradient(loss_value, model.trainable_variables)
         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
         return loss_value

     # Training loop
     total_time = 1500
     w_max = 1.5

     for i in range(total_time):
         t = i % T
         c = i % C
         w_max = max(1, w_max - 0.001)

         X_train_reshaped = X_train.reshape((X_train.shape[0], 28, 28))
         X_train_tf = tf.convert_to_tensor(X_train_reshaped, dtype=tf.float32)
         Y_train_tf = tf.convert_to_tensor(y_train_one_hot, dtype=tf.float32)

         loss_value = step_CE(optimizer, model, X_train_tf, Y_train_tf, t, c, w_max)

         if i % 100 == 0:
             acc = accuracy(model, X_train_tf, Y_train_tf)
             print(f"Step {i}, Loss: {loss_value.numpy()}, Accuracy: {acc.numpy() * 100}%")
     ```

#### 7. **Test Set Evaluation and Prediction:**
   - The trained model is evaluated on the test set, and predictions are made.
     ```python
     X_test_reshaped = X_test.reshape((X_test.shape[0], 28, 28))
     X_test_tf = tf.convert_to_tensor(X_test_reshaped, dtype=tf.float32)

     predictions = model(X_test_tf)
     predicted_labels = tf.argmax(predictions, axis=1).numpy()
     ```

#### 8. **Visualization:**
   - Visualization of a subset of test set images along with their true and predicted labels.
     ```python
     import matplotlib.pyplot as plt

     fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
     for i, ax in enumerate(axes):
         ax.imshow(X_test[i], cmap='gray')
         ax.set_title(f"True: {y_test[i]}\nPred: {predicted_labels[i]}", fontsize=8)
         ax.axis('off')

     plt.show()
     ```

#### 9. **Training Loss History Plotting:**
   - Plotting the training loss history for visualization.
     ```python
     training_loss_history = [44.61333453655243, 95.42666673660278, ...]
     plt.figure(figsize=(12, 4))
     plt.subplot(1, 2, 1)
     plt.plot(training_loss_history, label='Training Loss', color='blue')
     plt.title('Training Loss')
     plt.xlabel('Training Step')
     plt.ylabel('Loss')
     plt.xticks(np.arange(0, len(training_loss_history), step=100), labels=np.arange(0, len(training_loss_history)//100))
     plt.legend()
     ```

This documentation provides an overview of the code structure, key functions, and their purposes in implementing a dynamical loss function during neural network training.
