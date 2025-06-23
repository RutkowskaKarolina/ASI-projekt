"""
This is a boilerplate pipeline 'model_validation'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    comprehensive_model_evaluation,
    generate_confusion_matrix,
    generate_classification_report,
    analyze_model_size,
    final_model_assessment
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=comprehensive_model_evaluation,
                inputs=["test_df"],  # ← tylko test_df
                outputs="performance_metrics",
                name="comprehensive_model_evaluation_node",
            ),
            node(
                func=generate_confusion_matrix,
                inputs="performance_metrics",
                outputs="confusion_matrix_path",
                name="generate_confusion_matrix_node",
            ),
            node(
                func=generate_classification_report,
                inputs="performance_metrics",
                outputs="classification_report_path",
                name="generate_classification_report_node",
            ),
            node(
                func=analyze_model_size,
                inputs=None,  # ← nie przekazujemy nic
                outputs="model_size_analysis",
                name="analyze_model_size_node",
            ),
            node(
                func=final_model_assessment,
                inputs="performance_metrics",
                outputs="final_assessment",
                name="final_assessment_node",
            ),
        ]
    )
