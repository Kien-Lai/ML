import mnist

Images = mnist.read_idx('./MNIST_data/t10k-images.idx3-ubyte')
Labels = mnist.read_idx('./MNIST_data/t10k-labels.idx1-ubyte')

print(Images[0])
print(Labels[0])