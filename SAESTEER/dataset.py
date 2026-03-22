"""
Dataset loaders for SAE steering experiments.
"""

from datasets import load_dataset, Dataset


DATASET_IDS = {
    "sentiment": "Zirui22Ray/sentiment-dataset",
    "truthfulness": "wwbrannon/TruthGen",
    "politics": "Zirui22Ray/politics-dataset-demo",
}


def load_sentiment(test_size=0.2, seed=42):
    """Load sentiment dataset. Returns (train, test, positive_texts, negative_texts)"""
    ds = load_dataset(DATASET_IDS["sentiment"])
    rows = []
    for item in ds["train"]:
        if item["label"] == 'positive':
            rows.append({"text": item["text"], "label": 1})
        else:
            rows.append({"text": item["text"], "label": 0})
    unified = Dataset.from_list(rows)
    split = unified.train_test_split(test_size=test_size, seed=seed)
    train, test = split["train"], split["test"]
    positive = [x["text"] for x in train if x["label"] == 1]
    negative = [x["text"] for x in train if x["label"] == 0]

    return train, test, positive, negative


def load_truthfulness(test_size=0.2, seed=42):
    """Load TruthGen dataset. Returns (train, test, true_texts, false_texts)"""
    ds = load_dataset(DATASET_IDS["truthfulness"])
    rows = []                                                                                                                                
    for item in ds["train"]:                                                                                                                 
        rows.append({"text": item["truth"],     "label": 1})                                                                                 
        rows.append({"text": item["falsehood"],  "label": 0})   
    unified = Dataset.from_list(rows)

    split = unified.train_test_split(test_size=test_size, seed=seed)
    train, test = split["train"], split["test"]

    true_texts = [x["text"] for x in train if x["label"] == 1]
    false_texts = [x["text"] for x in train if x["label"] == 0]



    return train, test, true_texts, false_texts


def load_politics(test_size=0.2, seed=42):
    """Load politics dataset. Returns (train, test, right_texts, left_texts)"""
    ds = load_dataset(DATASET_IDS["politics"])
    
    split = ds["train"].train_test_split(test_size=test_size, seed=seed)
    train, test = split["train"], split["test"]

    right = [x["text"] for x in train if x["label"] == 1]
    left = [x["text"] for x in train if x["label"] == 0]
   

    return train, test, right, left


