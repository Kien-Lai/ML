import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
import pickle
import mnist

X_origin = mnist.read_idx('./MNIST_data/train-images.idx3-ubyte')

kmeans = KMeans(n_clusters=19)
kmeans.fit(X_origin.reshape(60000,784))
# kmeans.init = X_origin[0:10,0:28,0:28].reshape(10,784)

centroids = kmeans.cluster_centers_
lables = kmeans.labels_

print(centroids)
print(lables)

filename = 'model_19.sav'
pickle.dump(kmeans, open(filename, 'wb'))
