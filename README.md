# Soil Type Classification and Crop Recommendation System 🌱

A deep learning-based web application that identifies soil types from images and recommends suitable crops for cultivation. The system achieves an accuracy of 83.3% in soil type classification.

## 🚀 Features

- Image-based soil type classification
- Automatic crop recommendations based on soil type
- User-friendly web interface built with Streamlit
- Supports 5 different soil types:
  - Black Soil
  - Cinder Soil
  - Laterite Soil
  - Peat Soil
  - Yellow Soil

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/soil-classification.git
cd soil-classification
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Try the Live Demo

You can try the application directly through our Streamlit deployment:
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

### Local Development

1. Start the Streamlit application locally:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload an image of soil through the web interface

4. The system will:
   - Display the uploaded image
   - Classify the soil type
   - Provide crop recommendations based on the identified soil type

## 📁 Project Structure

- `app.py` - Main Streamlit application
- `soil_classifier_model.h5` - Trained deep learning model
- `SOIL.ipynb` - Jupyter notebook containing model training code
- `s0il.py` - Additional utility functions
- `Soil types/` - Directory containing training data
- `requirements.txt` - Project dependencies

## 🧪 Model Details

- Model Architecture: Deep Learning (Convolutional Neural Network)
- Input Size: 150x150 pixels
- Output Classes: 5 soil types
- Accuracy: 83.3%

## 🌾 Crop Recommendations

The system provides specific crop recommendations for each soil type:

- **Black Soil**: Cotton, Soybean, Sorghum, Maize, Sunflower, Millets, Pulses
- **Cinder Soil**: Millets, Oilseeds, Pulses (used in hilly/volcanic areas with proper irrigation)
- **Laterite Soil**: Tea, Coffee, Cashew, Coconut, Tapioca, Pineapple
- **Peat Soil**: Rice (Paddy), Potatoes, Sugar Beet, Vegetables
- **Yellow Soil**: Groundnut, Maize, Cotton, Pulses, Oilseeds

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
