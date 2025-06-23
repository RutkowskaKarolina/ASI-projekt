"""
Model validation pipeline for Indian food classification.
"""

import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from autogluon.multimodal import MultiModalPredictor

def comprehensive_model_evaluation(test_df, model_path='data/06_models/autogluon_model'):
    """Calculate comprehensive model evaluation metrics."""
    predictor = MultiModalPredictor.load(model_path)

    start_time = time.time()
    predictions = predictor.predict(test_df)
    inference_time = time.time() - start_time

    accuracy = accuracy_score(test_df['label'], predictions)
    precision = precision_score(test_df['label'], predictions, average='weighted')
    recall = recall_score(test_df['label'], predictions, average='weighted')
    f1 = f1_score(test_df['label'], predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'inference_time': inference_time,
        'avg_inference_time': inference_time / len(test_df),
        'predictions': predictions,
        'true_labels': test_df['label'].values,
        'predictor': predictor
    }


def generate_confusion_matrix(performance_metrics, save_path='data/08_reporting/confusion_matrix.png'):
    """Generate and save confusion matrix."""
    true = performance_metrics['true_labels']
    pred = performance_metrics['predictions']
    cm = confusion_matrix(true, pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Indian Food Classification')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path


def generate_classification_report(performance_metrics, save_path='data/08_reporting/classification_report.txt'):
    """Generate detailed classification report."""
    predictor = performance_metrics['predictor']
    true = performance_metrics['true_labels']
    pred = performance_metrics['predictions']
    categories = predictor.class_labels

    true_names = [categories[i] for i in true]
    pred_names = [categories[i] for i in pred]

    report = classification_report(true_names, pred_names, target_names=categories, digits=3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            f.write("INDIAN FOOD CLASSIFICATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)
    except Exception as e:
        print(f"[ERROR] Could not save classification report: {e}")

    return save_path


def analyze_model_size(model_path='data/06_models/autogluon_model', save_path='data/08_reporting/model_analysis.txt'):
    """Analyze model size on disk."""
    try:
        total_size = 0
        file_count = 0

        for root, _, files in os.walk(model_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_size += file_size
                file_count += 1

        size_mb = total_size / (1024 ** 2)
        size_gb = size_mb / 1024

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("MODEL SIZE ANALYSIS\n")
            f.write("=" * 25 + "\n")
            f.write(f"Total files: {file_count}\n")
            f.write(f"Model size: {size_mb:.2f} MB ({size_gb:.2f} GB)\n")

        return {'file_count': file_count, 'size_mb': size_mb, 'size_gb': size_gb}

    except Exception as e:
        print(f"[ERROR] Model size analysis failed: {e}")
        return None


def final_model_assessment(performance_metrics, save_path='data/08_reporting/final_assessment.txt'):
    """Provide final assessment for production readiness."""
    accuracy = performance_metrics['accuracy']
    avg_inference_time = performance_metrics['avg_inference_time']

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w') as f:
            f.write("FINAL MODEL ASSESSMENT\n")
            f.write("=" * 25 + "\n\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
            f.write(f"Avg Inference Time: {avg_inference_time:.3f} seconds\n")
    except Exception as e:
        print(f"[ERROR] Could not save final assessment: {e}")

    return {'accuracy': accuracy, 'avg_inference_time': avg_inference_time}


def train_model(train_df, valid_df, time_limit=3600, model_path="data/06_models/autogluon_model"):
    """Train the model and save it to disk."""
    predictor = MultiModalPredictor(
        label="label",
        path=model_path,
        problem_type="multiclass",
        eval_metric="accuracy"
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=valid_df,
        time_limit=time_limit,
        overwrite=True
    )

    predictor.save(model_path)
    print(f"Model saved to {model_path}")
    return None
