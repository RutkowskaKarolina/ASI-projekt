# 🍛 Indian Food Classifier

This project is a machine learning-powered web app that classifies Indian dishes from images. It can recognize 80 different types of dishes such as dosa, samosa, butter chicken, and more.

## Dataset
This project is based on the [Indian Food Images Dataset](https://www.kaggle.com/datasets/iamsouravbanerjee/indian-food-images-dataset) from Kaggle, containing 4,000 images across 80 Indian food classes.

## Live Demo
[View on Hugging Face Spaces](https://huggingface.co/spaces/ska24680/indian-food-classifier)

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
## Running the App
```bash
streamlit run app.py
```

## Running the ML Pipeline (Optional)
```bash
kedro run
```
