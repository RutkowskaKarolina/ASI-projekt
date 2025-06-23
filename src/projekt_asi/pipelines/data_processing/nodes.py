# Copyright 2024 Aradekin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.7
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np


def preprocess_data(parameters):
    """
    Scans the raw data directory, creates a DataFrame with image paths and labels,
    and splits it into training, validation, and test sets with improved data quality.
    """
    raw_data_path = 'data/01_raw/indian_food'
    image_files = []

    print("Scanning raw data directory...")

    for dirpath, _, filenames in os.walk(raw_data_path):
        category = os.path.basename(dirpath)
        valid_images = 0

        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(dirpath, filename)

                # Validate image can be opened
                try:
                    with Image.open(image_path) as img:
                        # Check if image is valid and has reasonable dimensions
                        if img.size[0] > 50 and img.size[1] > 50:  # Minimum size check
                            image_files.append({
                                "image": image_path,
                                "label": category
                            })
                            valid_images += 1
                except Exception as e:
                    print(f"Skipping invalid image {image_path}: {e}")
                    continue

        print(f"{category}: {valid_images} valid images")

    df = pd.DataFrame(image_files)

    if df.empty:
        raise ValueError("No valid images found in the raw data directory!")

    print(f"\n Dataset Summary:")
    print(f"   Total images: {len(df)}")
    print(f"   Categories: {df['label'].nunique()}")

    # Remove duplicates based on image path
    initial_count = len(df)
    df = df.drop_duplicates(subset=['image']).reset_index(drop=True)
    duplicates_removed = initial_count - len(df)
    print(f"   Duplicates removed: {duplicates_removed}")

    # Analyze class distribution
    class_counts = df['label'].value_counts()
    print(f"\n Class Distribution:")
    print(f"   Min samples per class: {class_counts.min()}")
    print(f"   Max samples per class: {class_counts.max()}")
    print(f"   Mean samples per class: {class_counts.mean():.1f}")
    print(f"   Std samples per class: {class_counts.std():.1f}")

    # Check for class imbalance
    imbalance_ratio = class_counts.max() / class_counts.min()
    if imbalance_ratio > 3:
        print(f"Class imbalance detected (ratio: {imbalance_ratio:.1f})")
    else:
        print(f"Class distribution is relatively balanced")

    # The new MultiModalPredictor expects label to be integer encoded
    df['label'] = pd.factorize(df['label'])[0]

    # First split to separate out the test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=parameters["model_options"]["test_size"],
        random_state=parameters["model_options"]["random_state"],
        stratify=df["label"]
    )

    # Second split to create training and validation sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=parameters["model_options"]["test_size"],
        random_state=parameters["model_options"]["random_state"],
        stratify=train_val_df["label"]
    )

    print(f"\n Final Split Summary:")
    print(f"   Training set: {len(train_df)} images")
    print(f"   Validation set: {len(val_df)} images")
    print(f"   Test set: {len(test_df)} images")

    return train_df, val_df, test_df