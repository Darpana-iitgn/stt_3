import csv
import json
from sklearn.metrics import cohen_kappa_score
from collections import Counter

def read_csv(file_path):
    annotators = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                annotations = json.loads(row['label'])

                for annotation in annotations:
                    start = annotation.get('start')  
                    end = annotation.get('end')
                    text = annotation.get('text')
                    labels = annotation.get('labels', []) 

                    if labels:  
                        labels = [label.lower() for label in labels]
                    else:
                        labels = ['no_label']  

                    annotators.append({
                        'start': start,
                        'end': end,
                        'text': text,
                        'labels': labels
                    })
            except json.JSONDecodeError:
                print(f"Error parsing JSON in row: {row}")
    return annotators

def calculate_cohen_kappa(annotations1, annotations2):
    
    annotator1_labels = [entry['labels'] for entry in annotations1]
    annotator2_labels = [entry['labels'] for entry in annotations2]

    min_length = min(len(annotator1_labels), len(annotator2_labels))
    annotator1_labels = annotator1_labels[:min_length]
    annotator2_labels = annotator2_labels[:min_length]

    annotator1_labels_flat = [label[0] for label in annotator1_labels]
    annotator2_labels_flat = [label[0] for label in annotator2_labels]

    kappa = cohen_kappa_score(annotator1_labels_flat, annotator2_labels_flat)
    
    
    print("Annotator 1 label distribution:", Counter([label for labels in annotator1_labels for label in labels]))
    print("Annotator 2 label distribution:", Counter([label for labels in annotator2_labels for label in labels]))

    return kappa

annotator1 = read_csv('nlp_darpana.csv')
annotator2 = read_csv('NLP_aeshaa.csv') 
kappa_score = calculate_cohen_kappa(annotator1, annotator2)

if kappa_score is not None:
    print(f"Cohen's Kappa: {kappa_score}")