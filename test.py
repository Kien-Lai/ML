import pickle
import mnist
import numpy as np
from collections import Counter

filename = 'model_14.sav'
kmeans = pickle.load(open(filename, 'rb'))
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

X = mnist.read_idx('./MNIST_data/train-labels.idx1-ubyte')

w, h = 14, 6000
list = [[0 for x in range(w)] for y in range(h)] 

for i in range(60000):
    list[labels[i]].append(X[i])

xyz = 0

for i in range(w):
    xyz = xyz + len(list[i])
    count = Counter(list[i])
    print("label ",i," is ",count.most_common())   

print(xyz)    