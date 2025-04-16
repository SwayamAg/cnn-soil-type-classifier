# 🌱 Soil Type Classification and Crop Recommendation System

A deep learning-based web application that classifies soil types from uploaded images and recommends suitable crops based on the predicted soil type. This system is built using a custom-trained CNN model and achieves **86% validation accuracy**.

---

## 🚀 Features

- 🔍 Image-based soil type classification using CNN
- 🌾 Automatic crop recommendations based on predicted soil
- 🖼️ Manual image upload and random prediction support
- 🧠 Streamlit-based interactive web interface
- ✅ Supports 5 specific soil types:
  - Black Soil
  - Cinder Soil
  - Laterite Soil
  - Peat Soil
  - Yellow Soil

> _Note: The model is trained to recognize only the above five soil types._

---

## 🧠 Model Summary

- **Model Type:** Convolutional Neural Network (CNN)
- **Input Shape:** 150x150 RGB images
- **Output Classes:** 5 soil types
- **Validation Accuracy:** 86%
- **Libraries Used:** Keras, TensorFlow, NumPy, Streamlit

---

## 📊 Training Performance

The following graphs were generated during the training phase using the training notebook:

### 📈 Accuracy vs Epochs
![Training Accuracy](assets/training_accuracy.png)

### 📉 Loss vs Epochs
![Training Loss](assets/training_loss.png)

> _You can find the training code and graphs in `SOIL.ipynb`._

---

## 📋 Requirements

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

```
tensorflow
keras
streamlit
numpy
pillow
```

---

## 🛠️ Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/soil-classifier.git
cd soil-classifier
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

### 4. Open in your browser

Visit `http://localhost:8501` to interact with the app.

---

## 📁 Project Structure

```
soil-classifier/
├── app.py                   # Streamlit app
├── s0il.py                  # Helper file for prediction
├── SOIL.ipynb               # Jupyter notebook for training
├── soil_classifier_model.h5 # Trained CNN model
├── requirements.txt         # Python dependencies
├── assets/                  # Graphs and images
│   ├── training_accuracy.png
│   └── training_loss.png
└── README.md
```

---

## 🌾 Crop Recommendations

Each soil type corresponds to specific crop suggestions:

- **Black Soil** → Cotton, Soybean, Sorghum, Maize, Sunflower, Millets, Pulses  
- **Cinder Soil** → Millets, Oilseeds, Pulses _(used in hilly/volcanic areas with proper irrigation)_  
- **Laterite Soil** → Tea, Coffee, Cashew, Coconut, Tapioca, Pineapple  
- **Peat Soil** → Rice (Paddy), Potatoes, Sugar Beet, Vegetables  
- **Yellow Soil** → Groundnut, Maize, Cotton, Pulses, Oilseeds

---

## 🌐 Deployment

You can deploy this app on **Streamlit Cloud** easily:

- Upload the project to a GitHub repo
- Go to [Streamlit Cloud](https://streamlit.io/cloud) and link your GitHub
- Set `app.py` as the entry point and provide the `requirements.txt`

---

## 🙌 Contributing

Feel free to fork this project, suggest improvements, or raise issues. Contributions are welcome!

---

## 📝 License

This project is licensed under the **MIT License**.

---

## 📞 Contact

Have feedback or questions? Feel free to open an issue in the repository.

---

_Developed with ❤️ using Deep Learning and Streamlit_
