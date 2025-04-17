import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 📌 Basic Settings
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 8
EPOCHS = 15
DATASET_PATH = "Soil types/"
MODEL_PATH = "soil_classifier_model.h5"

# 1️⃣ Data Augmentation & Preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# 2️⃣ Train and Validation Generators
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# 🔖 Get Class Labels
class_names = list(train_generator.class_indices.keys())
print("📌 Classes:", class_names)

# ✅ Optional: Visualize Augmented Images
images, _ = next(train_generator)
plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i])
    plt.axis('off')
plt.suptitle("📸 Sample Augmented Images")
plt.tight_layout()
plt.show()

# 3️⃣ Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(train_generator.num_classes, activation='softmax')
])

# 4️⃣ Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 🧠 EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# 5️⃣ Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# 💾 Save the trained model
model.save(MODEL_PATH)
print(f"✅ Model saved to: {MODEL_PATH}")

# 6️⃣ Accuracy & Loss Plots
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('📈 Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('📉 Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()

# 🧪 Classification Report
val_generator.reset()
preds = model.predict(val_generator, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_generator.classes

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("🧾 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

val_loss, val_acc = model.evaluate(val_generator, verbose=1)
print(f"\n📋 Overall Validation Accuracy: {val_acc * 100:.2f}%")

# 🌱 Updated Soil to Crop Recommendation Dictionary
soil = {
    "Black Soil": "🌾 Suitable crops: Cotton, Soybean, Sorghum, Maize, Sunflower, Millets, Pulses",
    "Cinder Soil": "🌾 Suitable crops: Millets, Oilseeds, Pulses (used in hilly/volcanic areas with proper irrigation)",
    "Laterite Soil": "🌾 Suitable crops: Tea, Coffee, Cashew, Coconut, Tapioca, Pineapple",
    "Peat Soil": "🌾 Suitable crops: Rice (Paddy), Potatoes, Sugar Beet, Vegetables",
    "Yellow Soil": "🌾 Suitable crops: Groundnut, Maize, Cotton, Pulses, Oilseeds"
}

# 🔮 Predict Any Image
def predict_soil_type_from_image(image_path):
    try:
        # Load model before prediction
        if not os.path.exists(MODEL_PATH):
            print("❌ Saved model not found!")
            return
        loaded_model = load_model(MODEL_PATH)

        img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = loaded_model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        raw_label = class_names[predicted_class_index]
        formatted_label = raw_label.replace('_', ' ').strip().title()

        # Show image + prediction
        plt.figure()
        plt.imshow(load_img(image_path))
        plt.axis('off')
        plt.title(f"📢 Predicted: {formatted_label}")
        plt.show()

        print(f"📢 Predicted: {formatted_label}")

        matched_soil = next((key for key in soil if key.lower() == formatted_label.lower()), None)
        if matched_soil:
            print(f"🌾 {soil[matched_soil]}")
        else:
            print("⚠️ No crop data available for this soil type.")
    except Exception as e:
        print(f"❌ Error: {e}")

# 🔧 Manually Predict Using a Custom Image Path
def manual_predict():
    path = input("📂 Enter the path of the soil image: ").strip()
    if os.path.exists(path):
        predict_soil_type_from_image(path)
    else:
        print("❌ The file path you entered doesn't exist.")

# 🎲 Predict Random Image from Dataset
def predict_random_image():
    print("\n🔍 Testing Random Image from Dataset...")
    random_class = random.choice(class_names)
    class_path = os.path.join(DATASET_PATH, random_class)
    image_name = random.choice(os.listdir(class_path))
    image_path = os.path.join(class_path, image_name)
    predict_soil_type_from_image(image_path)

# 🔽 CALL PREDICTION FUNCTIONS HERE
manual_predict()        # 🔍 Enable for manual path input prediction
predict_random_image()  # 🎲 Enable for random image prediction
