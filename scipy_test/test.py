import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data = pd.read_csv("dataset_test.csv", header=None)
print(data)
centroids = np.array([
[1000.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [3.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [4.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [5.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [6.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0],
        [7.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]])
model = KMeans(n_clusters=8, init=centroids, max_iter=1000, algorithm="full", verbose=0)
model.fit(data)
print(model.labels_)
data["labels"] = model.predict(data)
#print(data["labels"].values)
#print(data.groupby("labels").mean())
print(model.cluster_centers_)
print(data["labels"].value_counts())
