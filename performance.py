import pickle
import mnist

list = [0 for x in range(20)] 
list[0] = 9
list[1] = 8
list[2] = 2
list[3] = 1
list[4] = 0
list[5] = 6
list[6] = 6
list[7] = 5
list[8] = 7
list[9] = 2
list[10] = 5
list[11] = 3
list[12] = 9
list[13] = 0
list[14] = 1
list[15] = 8
list[16] = 3
list[17] = 4
list[18] = 7
list[19] = 4

filename = 'model.sav'
kmeans = pickle.load(open(filename, 'rb'))
labels = kmeans.labels_

Images = mnist.read_idx('./MNIST_data/t10k-images.idx3-ubyte')
Labels = mnist.read_idx('./MNIST_data/t10k-labels.idx1-ubyte')

result = kmeans.predict(Images.reshape(10000,784))

count = 0
for i in range(10000):
    if list[result[i]] == Labels[i]:
        count = count + 1

print(count)        