import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import os
from keras.optimizers import Adam

# Load the data
data_dir = './data/dataset/'
x_data = np.load(os.path.join(data_dir, 'x_data.npy'))
y_data = np.load(os.path.join(data_dir, 'y_data.npy'))

# Check and reshape data to ensure it has the right dimensions
num_samples, height, width, channels = x_data.shape

# One-hot encode y_data if necessary
y_data = tf.keras.utils.to_categorical(y_data, num_classes=5)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, shuffle=True, random_state=42)

# Define the model for static gesture recognition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu',padding = 'valid ',input_shape=(height, width, channels)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu',padding = 'valid'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu',padding = 'valid'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')  # Output layer for 3 classes
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 6
batch_size = 16
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f'{accuracy * 100:.2f}% of samples were classified correctly!')

# Save the trained model
model.save(os.path.join(data_dir, 'static_gesture_model.h5'))
