from scipy import misc
import numpy as np
import pickle
import mnist
import matplotlib.pyplot as plt

filename = 'model_20_init.sav'
X_test = mnist.read_idx('./MNIST_data/t10k-images.idx3-ubyte')
kmeans = pickle.load(open(filename, 'rb'))

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

def average(pixel):
    return (0.299*pixel[0] + 0.587*pixel[1] + 0.113*pixel[2])


def getClusterValue(id, clusterVals):
    return clusterVals[id]


def predictFromTest(id, clusterVals):
    try:
        id_num = int(id)
        clusterValue = kmeans.predict([X_test[id_num].reshape(784)])

        plt.gray()
        plt.imshow(X_test[id_num])
        plt.show()

        print("Ket qua: ", getClusterValue(clusterValue[0], clusterVals))
    except Exception as e:
        print(e)
        return print("Vui long nhap so")


while 1:
    id = input("Input id: ")
    if id == '':
        break
    predictFromTest(id, list)
