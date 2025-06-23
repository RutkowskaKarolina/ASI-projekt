import pandas as pd
import pickle
import os

train_df = pd.read_csv("data/02_intermediate/train_data.csv")

# Make a map: label number → class name (extracted from path)
label_map = {}
for _, row in train_df.iterrows():
    label = int(row['label'])
    path = row['image']
    class_name = os.path.basename(os.path.dirname(path))
    label_map[label] = class_name

label_map = dict(sorted(label_map.items()))

label_map_path = "data/06_models/label_map.pkl"
with open(label_map_path, "wb") as f:
    pickle.dump(label_map, f)

print(f"✅ Saved label_map to: {label_map_path}")
print("Label map:")
for k, v in label_map.items():
    print(f"{k}: {v}")
