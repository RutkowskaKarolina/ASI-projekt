"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_model, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train_model,
                inputs=["train_df", "val_df", "parameters"],
                outputs=None,
                name="train_model_node",
            ),
            node(
                func=evaluate_model,
                inputs=["test_df"],
                outputs="test_accuracy",
                name="evaluate_model_node",
            ),
        ]
    )
