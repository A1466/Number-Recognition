#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Import Libraries:

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


# In[15]:


#Load MNSIT Dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()


# In[16]:


#PreProcess the Data
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)


# In[17]:


#Build the Neural Network Model:
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[18]:


#Compile the Model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[19]:


#Train the Model
model.fit(train_images, train_labels, epochs=5, batch_size=64)


# In[20]:


# Evaluate the Model:
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')


# In[21]:


#Make Predictions
predictions = model.predict(test_images)


# In[22]:


#visulaization
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')
    plt.title(f'Predicted: {tf.argmax(predictions[i])}, Actual: {test_labels[i]}')
    plt.axis('off')
plt.show()


# In[ ]:





# In[ ]:




