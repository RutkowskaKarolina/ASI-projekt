# 🍛 Indian Food Classifier

[![Open in HF Spaces](https://img.shields.io/badge/🤗_Open-in_HF_Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/ska24680/indian-food-classifier)

This project is a machine learning-powered web app that classifies Indian dishes from images with 79.88% accuracy. It can recognize 80 different types of dishes such as dosa, samosa, butter chicken, and more.

## Dataset
This project is based on the [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) from Kaggle, containing 4,000 images across 80 Indian food classes.

## Preview

Here's a preview of the Indian Food Classifier UI built with Streamlit:

![App Screenshot](example/img.png)

## Tech Stack
- **Kedro** – Data pipelines and project structure  
- **AutoGluon** – Automated ML model training  
- **Streamlit** – Web UI for uploading images  
- **Docker** – Containerization  
- **Hugging Face Spaces** – Deployment and hosting  

## Installation

```bash
git clone https://github.com/RutkowskaKarolina/ASI-projekt.git
cd ASI-projekt
pip install -r requirements.txt
```
## Running the App locally
```bash
streamlit run streamlit_app.py
```

## Running the ML Pipeline (Optional)
```bash
kedro run
```
## Model Training

The model was trained using the Kedro pipeline with AutoGluon for image classification.  
To retrain the model from scratch:

```bash
kedro run --pipeline model_training
```

## Data and Model Setup
Due to repository size limitations, the dataset and trained model are not included in this GitHub repository. 

Download the trained model from Google Drive: [Trained Model](https://drive.google.com/drive/folders/1PEKBdiWPWfBkbp3WwfLwhl2dPrDHHlw6?usp=sharing)

Download the dataset from Kaggle: [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset)

## Project Data Structure (Kedro-style)
```bash
data/
├── 01_raw/                    # Place your dataset here
└── 06_models/                 # The resulting trained_model_predictor.pkl file will be saved here
    └── trained_model_predictor.pkl
```

## Unzip the contents into the folder:
data/01_raw/indian_food/
