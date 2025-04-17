# 🩺 Multi-Cancer Detection & Medical Analysis App

A deep learning-powered web app to detect cancer types from medical images using a CNN model, with AI-assisted analysis powered by Google's Gemini API. Built with TensorFlow, Streamlit, and Generative AI for educational and diagnostic support.

## 🚀 Features
- ✅ Upload medical images for cancer type detection
- ✅ Deep learning model trained on the Multi-Cancer dataset
- ✅ Transfer learning with VGG16
- ✅ Generates simplified medical analysis via Gemini API
- ✅ User-friendly interface with Streamlit

## 📁 Dataset
- **Source**: [Multi Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/obulisainaren/multi-cancer)
- **Content**: Labeled images of various cancer types
- **Preprocessing**: Images are resized to 224x224 and normalized

## 🧠 Model Architecture
- **Base Model**: VGG16 (pre-trained, layers frozen)
- **Custom Layers**:
  - GlobalAveragePooling2D
  - Dense Layer with ReLU
  - Dropout Layer
  - Dense Softmax Output Layer
- **Loss Function**: `SparseCategoricalCrossentropy`
- **Optimizer**: `Adam`
- **Epochs**: 5 (modifiable)

## 🛠️ Setup Instructions
### 1. Clone the Repository
```bash
git clone https://github.com/teja85053/Multi_cancer_diagnosis_and_Screening.git
cd multi-cancer-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the project root with your Gemini API key:
```
GEMINI_API_KEY=your_google_generativeai_key_here
```

### 4. Download Dataset
Download and extract the dataset from Kaggle into:
```bash
./data/Multi_Cancer/
```

## 🧪 Train the Model
Run the training script to prepare the model:
```bash
python train_model.py
```

This will:
- Train the model using the dataset
- Save it as `multi_cancer_model.h5` under `models/`
- Save class labels to `class_names.json` under `config/`

## 🌐 Run the Web App
Launch the Streamlit app:
```bash
streamlit run app.py
```

Functionality:
- Upload image files (.jpg, .jpeg, .png)
- Predict the cancer type using CNN
- Generate an understandable medical report using Gemini

## 📦 Project Structure
```
.
├── app.py                      # Streamlit application
├── train_model.py              # Model training script
├── models/
│   └── multi_cancer_model.h5   # Trained CNN model
├── config/
│   └── class_names.json        # Cancer class labels
├── data/
│   └── Multi_Cancer/           # Image dataset
├── .env                        # API key configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

## ✅ Requirements
- Python 3.7+
- TensorFlow
- Streamlit
- Google Generative AI SDK
- NumPy
- PIL
- dotenv

Install all with:
```bash
pip install -r requirements.txt
```

## 📌 Notes
- Do not push large files (e.g., datasets or models) to GitHub. Use .gitignore or Git LFS.
- Ensure your Gemini API is correctly configured in `.env`.
- For real-world medical use, ensure the model is validated with certified datasets.

## 🔐 Environment Variables
Create a `.env` file:
```
GEMINI_API_KEY=your_google_generativeai_key_here
```

## 🙏 Acknowledgements
- Kaggle: Multi Cancer Dataset
- TensorFlow
- Streamlit
- Google Gemini API

## 📄 License
This project is licensed for educational and research purposes only. Not for clinical or commercial deployment without proper validation and approval.
