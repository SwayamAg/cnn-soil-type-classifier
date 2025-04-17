# CNN Soil Type Classifier 🌱

A deep learning-based web application that identifies soil types from images and recommends suitable crops for cultivation. The system achieves an accuracy of 83.3% in soil type classification.

## 🚀 Features

- Image-based soil type classification using CNN
- Automatic crop recommendations based on soil type
- User-friendly web interface built with Streamlit
- Supports 5 different soil types:
  - Black Soil
  - Cinder Soil
  - Laterite Soil
  - Peat Soil
  - Yellow Soil

## 📋 Prerequisites

- Python 3.7+
- Git
- Google Drive account (for accessing the model)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnn-soil-type-classifier.git
cd cnn-soil-type-classifier
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 📥 Accessing the Model

The trained model is available on Google Drive. Follow these steps to download and set up the model:

1. Download the model file:
   - Click on this link: [Download Model](https://drive.google.com/drive/folders/1n0oR9dz6a_AEsHSuSPyHdb-ibs1eNGv_?usp=sharing)
   - The file will be named `soil_classifier_model.h5`

2. Place the model file:
   - Move the downloaded `soil_classifier_model.h5` file to the root directory of the project
   - The file structure should look like this:
     ```
     cnn-soil-type-classifier/
     ├── app.py
     ├── soil_classifier_model.h5
     ├── requirements.txt
     └── ...
     ```

3. Verify the model:
   - The model file should be approximately 55MB in size
   - Make sure the file is not corrupted during download

## 🎯 Usage

### Local Deployment

1. Make sure you have downloaded the model file from Google Drive and placed it in the project root directory

2. Start the Streamlit application locally:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

4. Upload an image of soil through the web interface

5. The system will:
   - Display the uploaded image
   - Classify the soil type
   - Provide crop recommendations based on the identified soil type

## 📁 Project Structure

- `app.py` - Main Streamlit application
- `soil_classifier_model.h5` - Trained deep learning model (download from Google Drive)
- `SOIL.ipynb` - Jupyter notebook containing model training code
- `s0il.py` - Additional utility functions
- `requirements.txt` - Project dependencies

## 🧪 Model Details

- Model Architecture: Convolutional Neural Network (CNN)
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

## 🙏 Acknowledgments

- Thanks to all contributors and maintainers
- Special thanks to the open-source community for their valuable tools and libraries
- Thanks to Streamlit for providing an excellent platform for deploying ML applications

## 📞 Contact

For any queries or suggestions, please open an issue in the repository. 