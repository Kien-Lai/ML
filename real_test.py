from scipy import misc
import numpy as np
import pickle
filename = 'model_20_init.sav'


kmeans = pickle.load(open(filename, 'rb'))

def average(pixel):
    return (0.299*pixel[0] + 0.587*pixel[1] + 0.113*pixel[2])

import matplotlib.pyplot as plt

# print (grey)

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

def getClusterValue(id, clusterVals):
    return clusterVals[id]

def predict(file, clusterVals):
    try:
        image = misc.imread(file)
        grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
        # get row number
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                grey[rownum][colnum] = average(image[rownum][colnum])
        clusterValue = kmeans.predict([grey.reshape(784)])

        print("Ket qua: ", getClusterValue(clusterValue[0], clusterVals))
    except Exception as e:
        return print("Vui long nhap dung: [path].png")

while 1:
    file = input("Input your image link: ")
    if file == '':
        break
    predict(file, list)
