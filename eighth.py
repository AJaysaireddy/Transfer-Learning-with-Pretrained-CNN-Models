import sys
sys.stdout.reconfigure(encoding='utf-8')

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the number of classes in your dataset
num_classes = 2  # Update this with the actual number of classes in your dataset

# Step 3: Model Architecture
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = models.Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Step 4: Transfer Learning
for layer in base_model.layers:
    layer.trainable = False

# Step 6: Evaluation Metrics
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Step 2: Dataset Preparation
train_dir = r'C:\Users\ajays\OneDrive\Desktop\online\assign4\archive (1)\alien_vs_predator_thumbnails\data\train'
validation_dir = r'C:\Users\ajays\OneDrive\Desktop\online\assign4\archive (1)\alien_vs_predator_thumbnails\data\validation'
batch_size = 32
num_epochs = 10

# Step 5: Data Augmentation
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical')

# Train the model
model.fit(train_generator, epochs=num_epochs, validation_data=validation_generator)
