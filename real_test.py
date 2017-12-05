from scipy import misc
import numpy as np
image = misc.imread('7.png')
import pickle
filename = 'model_20_init.sav'
kmeans = pickle.load(open(filename, 'rb'))

def average(pixel):
    return pixel[2]

grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
# get row number
for rownum in range(len(image)):
   for colnum in range(len(image[rownum])):
      grey[rownum][colnum] = average(image[rownum][colnum])

print (grey)      

print(kmeans.predict([grey.reshape(784)]))      