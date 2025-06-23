"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.7
"""

import os
import shutil
from autogluon.multimodal import MultiModalPredictor
from sklearn.metrics import accuracy_score

def train_model(train_df, val_df, parameters):
    """
    Trains an AutoGluon MultiModalPredictor model for image classification with maximum quality settings.
    """
    model_output_path = 'data/06_models/autogluon_model'
    model_options = parameters["model_options"]

    if os.path.exists(model_output_path):
        shutil.rmtree(model_output_path)

    print(f"Starting model training with MAXIMUM QUALITY configuration:")
    print(f"   - Time limit: {model_options['time_limit']} seconds ({model_options['time_limit']/3600:.1f} hours)")
    print(f"   - Preset: {model_options['presets']} (maximum quality)")
    print(f"   - This will use the best possible model architecture and training strategy")

    predictor = MultiModalPredictor(
        label="label",
        path=model_output_path,
        eval_metric="accuracy",
        verbosity=2,
        problem_type="multiclass"
    )

    # Train the model with maximum quality settings
    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        time_limit=model_options["time_limit"],
        presets=model_options["presets"]
    )

    print(f"Model training completed successfully!")
    print(f"Model saved to: {model_output_path}")

    predictor.save(model_output_path)
    return None

def evaluate_model(test_df):
    """
    Evaluates the trained model on the test set.
    """
    model_path = "data/06_models/autogluon_model"
    predictor = MultiModalPredictor.load(model_path)
    predictions = predictor.predict(test_df)
    accuracy = accuracy_score(test_df['label'], predictions)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
