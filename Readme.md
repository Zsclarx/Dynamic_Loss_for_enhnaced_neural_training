Name – Prakhar Pratap Singh 

Roll No. – 22075102 and 22074024 

Group No. – 63 (Prakhar Pratap Singh and Pratham Agarwal) 

**Google Colab link :-**  

[https://colab.research.google.com/drive/1fNNzxhwEc4jdvBdQa-Xj_DnUs3ENxrKP?usp=sharing](https://colab.research.google.com/drive/1fNNzxhwEc4jdvBdQa-Xj_DnUs3ENxrKP?usp=sharing)

The DynamicLoss function is designed to be a flexible and customizable loss function in TensorFlow/Keras. Let's break down each part of the DynamicLoss class: 

### Dynamical Loss Function Implementation Documentation:

#### 1. **Load and Preprocess Data:**
   - The code loads the MNIST dataset and preprocesses the images and labels.

#### 2. **Neural Network Architecture:**
   - The neural network architecture is defined using Keras. It's a simple feedforward network.

#### 3. **Dynamical Loss Function (Weighted Loss):**
   - A custom dynamical loss function `weighted_loss` is defined. The weights for the loss function are determined by the function `c_fn`.

     **Formulas:**
     - The dynamical weight function \( c_{\text{fn}}(t, i, w_{\text{max}}) \) is defined as follows:
       \[ w(t, i) = 1 + t \cdot \text{slope} \ \ \text{for} \ t < \frac{T}{2} \]
       \[ w(t, i) = 2 \cdot w_{\text{max}} - t \cdot \text{slope} - 1 \ \ \text{for} \ t \geq \frac{T}{2} \]
       \[ \text{slope} = \frac{2 \cdot (w_{\text{max}} - 1)}{T} \]
       \[ w = \frac{1}{C} \cdot \text{normalize}\left(\mathbb{1} + (w_{\text{main\_class}} - 1) \cdot \text{one\_hot}(i, C)\right) \]
       Where:
       - \( C \) is the number of classes.
       - \( \mathbb{1} \) is a vector of ones.
       - \( \text{one\_hot}(i, C) \) is a one-hot encoded vector for class \( i \).
       - \( w_{\text{main\_class}} \) is the main class weight.
       - The final weight vector \( w \) is normalized to ensure that its elements sum to \( C \).

#### 4. **Standard Loss Function:**
   - A standard cross-entropy loss function is defined for comparison during training.

     **Formula:**
     \[ \text{Standard Cross-Entropy Loss} = -\frac{1}{C} \sum_{j=1}^{C} y_j \cdot \log(\text{softmax}(\text{predictions})) \]
     Where:
     - \( C \) is the number of classes.
     - \( y_j \) is the one-hot encoded ground truth label for class \( j \).
     - \( \text{softmax}(\text{predictions}) \) is the softmax activation of the neural network predictions.

#### 5. **Accuracy Calculation:**
   - A function to calculate accuracy is defined.

     **Formula:**
     \[ \text{Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \text{I}(\text{true\_label}_i = \text{predicted\_label}_i) \]
     Where:
     - \( N \) is the number of samples.
     - \( \text{I}(condition) \) is the indicator function.

#### 6. **Training Loop:**
   - The training loop uses the custom loss function and updates the model's weights accordingly.

     **Formulas:**
     - The training loop iterates over time steps and updates the weights dynamically based on the current time \( t \) and class index \( c \).

This documentation provides a detailed overview of the code, including formulas for the dynamical loss function, standard loss function, accuracy calculation, and explanations of key components.
