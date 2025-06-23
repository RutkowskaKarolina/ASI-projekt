"""Project pipelines."""
from __future__ import annotations

from typing import Dict
from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline
from projekt_asi.pipelines import data_processing as dp
from projekt_asi.pipelines import model_training as mt
from projekt_asi.pipelines import model_validation as mv


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = dp.create_pipeline()
    model_training_pipeline = mt.create_pipeline()
    model_validation_pipeline = mv.create_pipeline()

    return {
        "__default__": data_processing_pipeline + model_training_pipeline + model_validation_pipeline,
        "dp": data_processing_pipeline,
        "mt": model_training_pipeline,
        "mv": model_validation_pipeline,
        "model_validation": model_validation_pipeline,
        "full": data_processing_pipeline + model_training_pipeline + model_validation_pipeline,
    }
