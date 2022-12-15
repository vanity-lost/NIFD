import os
from tensorflow import keras
import pandas as pd
import numpy as np

def get_data():
    zip_file = keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")

    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"],
    )

    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
    )
    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}
    paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

    papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
    citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
    citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
    papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

    labels = np.concatenate((papers["subject"].to_numpy(), np.zeros((50,), dtype=np.int64)), axis=0).reshape(-1, 1)

    edges = citations[["source", "target"]].to_numpy()
    A = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for i, j in edges:
        A[i][j] += 1
    for i in range(len(labels)-50, len(labels)):
        A[i][i] += 1

    feature_names = set(papers.columns) - {"paper_id", "subject"}
    features = np.concatenate((papers.sort_values("paper_id")[feature_names].to_numpy(), np.zeros((50, len(feature_names)), dtype=np.int64)), axis=0)

    inputs = np.concatenate((labels, features, A), axis=1)

    X, y = np.array([i for i in range(len(labels))]), labels
    just_edges = edges.T

    return inputs, X, y, just_edges, features, labels
