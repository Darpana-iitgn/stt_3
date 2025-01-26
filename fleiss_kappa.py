import json
from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np

def get_image_identifier(image_path):
    return image_path[-7:]

file_to_annotator = {
    "cv_darpana.json": 1,
    "cv_third.json": 2,
    "CV_aeshaa.json": 3}

annotations = []
for file_path, annotator_id in file_to_annotator.items():
    with open(file_path, "r") as f:
        data = json.load(f)
        for annotation in data:
            annotation["image"] = get_image_identifier(annotation["image"])
            annotation["annotator"] = annotator_id
            annotations.append(annotation)

label_mapping = {}
for annotation in annotations:
    label = annotation["choice"]
    if label not in label_mapping:
        label_mapping[label] = len(label_mapping)

grouped_annotations = defaultdict(lambda: defaultdict(list))
for annotation in annotations:
    image = annotation["image"]
    annotator = annotation["annotator"]
    choice = annotation["choice"]
    grouped_annotations[image][annotator] = label_mapping[choice]

annotators = set(annotation["annotator"] for annotation in annotations)
images = list(grouped_annotations.keys())
annotation_matrix = []
for image in images:
    row = []
    for annotator in annotators:
        row.append(grouped_annotations[image].get(annotator, -1))
    annotation_matrix.append(row)

num_labels = len(label_mapping)
frequency_matrix = np.zeros((len(annotation_matrix), num_labels), dtype=int)
for i, row in enumerate(annotation_matrix):
    for label in row:
        if label != -1:
            frequency_matrix[i, label] += 1

kappa = fleiss_kappa(frequency_matrix)

print("Fleiss's Kappa:", kappa)