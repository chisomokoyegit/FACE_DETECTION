from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# Image data setup
train_dir = 'train'
test_dir = 'test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    batch_size=64,
    color_mode='grayscale',
    class_mode='categorical'
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotion classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Create checkpoints folder if not exists
os.makedirs("checkpoints", exist_ok=True)

# Callbacks
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=2,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath='checkpoints/best_face_emotionModel.h5',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

try:
    # Train model (max 15 epochs)
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=test_generator,
        callbacks=[early_stop, checkpoint]
    )

except KeyboardInterrupt:
    print("\n🛑 Training interrupted manually. Saving current model...")

finally:
    # Always save current model
    model.save("face_emotionModel.h5")
    print("✅ Model saved successfully as face_emotionModel.h5")

    # Plot accuracy
    if 'history' in locals():
        plt.plot(history.history['accuracy'], label='Train acc')
        plt.plot(history.history['val_accuracy'], label='Val acc')
        plt.legend()
        plt.title("Training Progress")
        plt.show()



