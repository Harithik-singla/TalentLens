import json

def load_label_maps(path="./data/processed"):
    with open(f"{path}/label2id.json") as f:
        label2id = json.load(f)
    with open(f"{path}/id2label.json") as f:
        id2label = json.load(f)
    return label2id, id2label