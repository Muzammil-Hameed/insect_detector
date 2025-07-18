from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data Preparation
datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

# Training and Validation Data
train_data = datagen.flow_from_directory(
    'Dataset',
    target_size=(128, 128),  # Target size matches the model's input
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    'Dataset',
    target_size=(128, 128),  # Target size matches the model's input
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Build the Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),  # Input shape matches the data generator
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_data.class_indices), activation='softmax')  # Number of insect types
])

# Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save the Model
model.save('insect_detector.h5')
print("Model training complete. Saved as 'insect_detector.h5'")
