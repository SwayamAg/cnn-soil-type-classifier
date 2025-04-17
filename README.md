# 🌱 CNN Soil Type Classifier – Deployed on Streamlit!

A deep learning-based web application that **identifies soil types from images** and **recommends suitable crops** for cultivation. The system achieves an **86% validation accuracy** across 5 soil classes.

### 🚀 [Try the Live App on Streamlit →](https://cnn-soil-type-classifier.streamlit.app/)  

---

## ✨ Features

- 🔍 CNN-based image classification for soil types  
- 🌾 Automatic crop recommendations based on predicted soil class  
- 🖥️ Interactive UI built with **Streamlit**  
- 🧠 Trained on augmented image dataset with EarlyStopping  
- ✅ Tested on both random and user-uploaded soil images  
- 🌍 Supports 5 soil types:
  - Black Soil
  - Cinder Soil
  - Laterite Soil
  - Peat Soil
  - Yellow Soil

---

## 📋 Prerequisites

- Python 3.7+
- Git

---

## 🛠️ Installation (for local use)

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/cnn-soil-type-classifier.git
cd cnn-soil-type-classifier
```

2. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Info

- 📐 **Input Size**: 150x150 pixels  
- 🧠 **Architecture**: Custom CNN  
- 🎯 **Output Classes**: 5 soil types  
- 📊 **Validation Accuracy**: 86%  
- ✅ Model file `soil_classifier_model.h5` is already included in this repository

---

## 🌾 Crop Recommendations

Based on soil type, the system provides:

- **Black Soil** → Cotton, Soybean, Sorghum, Maize, Sunflower, Millets, Pulses  
- **Cinder Soil** → Millets, Oilseeds, Pulses (ideal in hilly/volcanic areas)  
- **Laterite Soil** → Tea, Coffee, Cashew, Coconut, Tapioca, Pineapple  
- **Peat Soil** → Rice (Paddy), Potatoes, Sugar Beet, Vegetables  
- **Yellow Soil** → Groundnut, Maize, Cotton, Pulses, Oilseeds

---

## 📽️ Demo Video

🎥 **Watch the Project in Action:**  
[Click here to watch the demo video](Demo.mp4)

Includes:
- Image-based soil classification  
- Crop suggestion output  
- Streamlit web interface walkthrough

---

## ⚙️ Run Locally (Optional)

If you'd like to test locally instead of using the deployed app:

1. Clone the repo (includes model file)  
2. Run the app using Streamlit:
```bash
streamlit run app.py
```

3. Open your browser at `http://localhost:8501`

---

## 📁 Project Structure

- `app.py` – Main Streamlit app  
- `soil_classifier_model.h5` – Trained CNN model  
- `SOIL.ipynb` – Notebook used to train the model  
- `s0il.py` – Helper functions  
- `requirements.txt` – Dependencies  
- `Demo.mp4` – Project demo video  
- `Soil types/` – Dataset containing soil images

---

## 🤝 Contributing

Open to improvements! Feel free to fork the repo, create issues, or submit pull requests.

---

## 📝 License

Licensed under the **MIT License**. See the `LICENSE` file for more info.
