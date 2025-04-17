import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time

# Soil to crop recommendation mapping
soil = {
    "Black Soil": "🌾 Suitable crops: Cotton, Soybean, Sorghum, Maize, Sunflower, Millets, Pulses",
    "Cinder Soil": "🌾 Suitable crops: Millets, Oilseeds, Pulses (used in hilly/volcanic areas with proper irrigation)",
    "Laterite Soil": "🌾 Suitable crops: Tea, Coffee, Cashew, Coconut, Tapioca, Pineapple",
    "Peat Soil": "🌾 Suitable crops: Rice (Paddy), Potatoes, Sugar Beet, Vegetables",
    "Yellow Soil": "🌾 Suitable crops: Groundnut, Maize, Cotton, Pulses, Oilseeds"
}

# Set image size parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Load trained model
MODEL_PATH = "soil_classifier_model.h5"
if not os.path.exists(MODEL_PATH):
    st.error("Model not found!")
    st.stop()

model = load_model(MODEL_PATH)

# Streamlit app layout
st.title("Soil Type and Crop Recommendation")
st.markdown("""
    (This model can only predict the soil type and recommend crops for 5 specific soil types: 
    Black Soil, Cinder Soil, Laterite Soil, Peat Soil, Yellow Soil.)
""")

# Image upload section
uploaded_file = st.file_uploader("Choose an image of soil", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = load_img(uploaded_file, target_size=(IMG_HEIGHT, IMG_WIDTH))
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocessing the image
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Show loading spinner while predicting
    with st.spinner('Processing... Please wait while we analyze the image.'):
        # Simulate a slight delay (optional)
        time.sleep(2)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction)
        
        # Get the predicted soil type
        class_names = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']
        predicted_soil = class_names[predicted_class_index]
        
        # Display prediction and corresponding crop recommendations
        st.subheader(f"Predicted Soil Type: {predicted_soil}")
        st.write(f"Recommended Crop: {soil[predicted_soil]}")
