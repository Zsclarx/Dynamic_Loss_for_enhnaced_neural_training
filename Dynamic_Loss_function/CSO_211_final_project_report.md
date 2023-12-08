<a name="br1"></a> 

Name – Prakhar Pratap Singh

Roll No. – 22075102 and 22074024

Group No. – 63 (Prakhar Pratap Singh and Pratham Agarwal)

**Google Colab link  :-**

<https://colab.research.google.com/drive/1elILhdkU_eLoKOyvkO_beWpoh80MuidZ?usp=sharing>

The DynamicLoss function is designed to be a flexible and customizable loss function in TensorFlow/Keras.

Let's break down each part of the DynamicLoss class:

**Iniꢀalizaꢀon:**

def \_\_init\_\_(self, learning\_rate, curriculum\_schedule, \*\*kwargs):

super(DynamicLoss, self).\_\_init\_\_(\*\*kwargs)

self.learning\_rate = learning\_rate

self.curriculum\_schedule = curriculum\_schedule

self.epoch = 0 # To keep track of the current epoch

•

•

**learning\_rate**: This is a parameter that scales the contribution of the learning progress component in

the loss function.

**curriculum\_schedule**: This is a function that takes the current epoch as an argument and returns a

curriculum factor. The curriculum factor is used to scale the contribution of the curriculum loss

component.

•

**epoch**: This variable is used to keep track of the current epoch during training.

**Loss Calculation**:

def call(self, y\_true, y\_pred):

\# Custom loss components (modify as needed)

standard\_loss = tf.keras.losses.categorical\_crossentropy(y\_true, y\_pred)

learning\_progress\_loss = self.learning\_rate \* self.learning\_progress()

class\_balance\_loss = self.class\_balance\_loss(y\_true)

curriculum\_loss = self.curriculum\_loss(y\_pred)



<a name="br2"></a> 

dynamic\_loss = standard\_loss + learning\_progress\_loss + class\_balance\_loss +

curriculum\_loss

return dynamic\_loss

\* **standard\_loss**: This is the standard categorical cross-entropy loss between the true labels (y\_true) and

the predicted labels (y\_pred).

\* **learning\_progress\_loss**: This component represents a custom loss term that increases linearly with

epochs. It is scaled by the specified learning rate (self.learning\_rate).

\* **class\_balance\_loss**: This component penalizes class imbalance. It calculates the cross-entropy loss

between the average class distribution in y\_true and a uniform distribution.

\* **curriculum\_loss**: This component represents a curriculum-based loss term. It is based on the sine of

y\_pred values and is scaled by a curriculum factor obtained from the curriculum\_schedule function.

**The final dynamic\_loss is the sum of these individual components**.

**Learning Progress and Class Balance Loss Functions:**

def learning\_progress(self):

\# Example: Learning progress increases linearly with epochs

return self.epoch / num\_epochs # Adjust as needed

def class\_balance\_loss(self, y\_true):

\# Example: Penalize class imbalance

class\_distribution = tf.reduce\_mean(y\_true, axis=0)

return tf.keras.losses.categorical\_crossentropy(class\_distribution,

tf.ones\_like(class\_distribution) / len(class\_distribution))

•

**learning\_progress**: This function defines the learning progress, which increases linearly with

epochs. It is used in the calculation of the learning progress loss.

•

**class\_balance\_loss**: This function penalizes class imbalance by calculating the cross-entropy loss

between the average class distribution in y\_true and a uniform distribution.



<a name="br3"></a> 

**Curriculum Loss Function:**

def curriculum\_loss(self, y\_pred):

\# Example: Curriculum loss based on the sine of y\_pred values

curriculum\_factor = self.curriculum\_schedule(self.epoch)

return curriculum\_factor \* 0.1 \* tf.reduce\_sum(tf.abs(tf.math.sin(y\_pred)))

**curriculum\_loss** : This function calculates a curriculum-based loss term. It is based on the sine of y\_pred

values and is scaled by a curriculum factor obtained from the curriculum\_schedule function.



<a name="br4"></a> 

**Learning Progress Tracking:**

def on\_epoch\_end(self, epoch, logs=None):

self.epoch = epoch

•

on\_epoch\_end: This method updates the self.epoch variable at the end of each epoch, ensuring that

the learning progress and curriculum loss functions get the correct epoch information.

In summary, the DynamicLoss function is a versatile custom loss function that allows you to incorporate

various components into your neural network training process. It includes standard cross-entropy loss,

learning progress loss, class balance loss, and curriculum loss, each of which can be adjusted and

customized based on your specific requirements. The learning progress and epoch tracking ensure that the

dynamic components evolve over the course of training.

