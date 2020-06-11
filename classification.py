# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import tensorflow as tf


# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt


# %%
#Defining training and validation paths
PATH = os.path.join(os.getcwd())
train_dir = os.path.join(PATH, 'training')
validation_dir = os.path.join(PATH, 'validation')
print(train_dir)


# %%
#Defining dataset paths for taining and validation
train_cats_dir = os.path.join(train_dir, 'cats')  
train_dogs_dir = os.path.join(train_dir, 'dogs')  
validation_cats_dir = os.path.join(validation_dir, 'cats')  
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
print(os.listdir(train_cats_dir))


# %%
#Analysing the total dataset we have
num_cats_tr = len(os.listdir(train_cats_dir))
num_dogs_tr = len(os.listdir(train_dogs_dir))

num_cats_val = len(os.listdir(validation_cats_dir))
num_dogs_val = len(os.listdir(validation_dogs_dir))

total_train = num_cats_tr + num_dogs_tr
total_val = num_cats_val + num_dogs_val


# %%
print('total training cat images:', num_cats_tr)
print('total training dog images:', num_dogs_tr)

print('total validation cat images:', num_cats_val)
print('total validation dog images:', num_dogs_val)
print("--")
print("Total training images:", total_train)
print("Total validation images:", total_val)


# %%
#Defining variables for pre-processing
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# %% [markdown]
# ## Data Preparation
# %% [markdown]
# Format the images into appropriately pre-processed floating point tensors before feeding to the network:
# 
# 1. Read images from the disk.
# 2. Decode contents of these images and convert it into proper grid format as per their RGB content.
# 3. Convert them into floating point tensors.
# 4. Rescale the tensors from values between 0 and 255 to values between 0 and 1, as neural networks prefer to deal with small input values.
# 
# The ImageDataGenerator class provided by tf.keras. It can read images from disk and preprocess them into proper tensors. It will also set up generators that convert these images into batches of tensors—helpful when training the network.

# %%
#Generator for training data
train_image_generator = ImageDataGenerator(rescale=1./255)

#Generator for validation data
validation_image_generator = ImageDataGenerator(rescale=1./255)

# %% [markdown]
# The flow_from_directory method load images from the disk, applies rescaling, and resizes the images into the required dimensions.

# %%
#Pre=processing of training data
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')


# %%
#Pre=processing of validation data
val_data_gen = validation_image_generator.flow_from_directory
(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

# %% [markdown]
# ## Creating model
# %% [markdown]
# The model consists of three convolution blocks with a max pool layer in each of them. There's a fully connected layer with 512 units on top of it that is activated by a `relu` activation function.
# 
# The Rectified Linear Unit is the most commonly used activation function in deep learning models. The function returns 0 if it receives any negative input, but for any positive value  x  it returns that value back.

# %%
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])


# %%
#Compiling the model
model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])


# %%
model.summary()


# %%
#Training the model
history = model.fit(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)


# %%
#Ploting training, validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %%
#Testing the trained model
from os import listdir
from os.path import isfile,join
import numpy as np

#Preparing testing data
predict_dir_path='testing/'
test_files = [f for f in listdir(predict_dir_path) if isfile(join(predict_dir_path, f))]
print(test_files)

IMG_WIDTH=150
IMG_HEIGHT=150


# %%
#Performing pre-processing
from keras.preprocessing import image

dog_counter = 0 
cat_counter  = 0
for file in test_files:
    img = image.load_img(predict_dir_path+file, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    #Precdicting the input image  
    classes = model.predict_classes(images, batch_size=10)
    classes = classes[0][0]
    
    if classes == 0:
        print(file + ": " + 'cat')
        cat_counter += 1
    else:
        print(file + ": " + 'dog')
        dog_counter += 1

