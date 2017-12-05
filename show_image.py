import matplotlib.pyplot as plt 
import mnist

fig = plt.figure()
fig.patch.set_facecolor('white')

X = mnist.read_idx('./MNIST_data/t10k-images.idx3-ubyte')

print(X[0])

plt.imshow(X[0])
plt.show()