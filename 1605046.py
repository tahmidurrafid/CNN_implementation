from copy import deepcopy
from multiprocessing import Pool
import idx2numpy
import numpy
import cv2

train_images = 'mnist/train-images.idx3-ubyte'
train_labels = 'mnist/train-labels.idx1-ubyte'
train_images = idx2numpy.convert_from_file(train_images)
train_labels = idx2numpy.convert_from_file(train_labels)
train_images = train_images.reshape(-1, 28, 28, 1)

class Convolution:
    def __init__(self, filter_dim, in_dim, padding, stride):
        self.padding = padding
        self.stride = stride
        self.filter_count = filter_dim[3]
        self.filter_dim = filter_dim
        self.in_dim = in_dim
        self.out_dim = (in_dim[0] , (in_dim[1] + 2*self.padding - self.filter_dim[0])//self.stride + 1, \
            (in_dim[2] + 2*self.padding - self.filter_dim[1])//self.stride  + 1, self.filter_count)
        self.bias = numpy.random.randn(self.filter_count)
        self.filters = numpy.random.randn(*filter_dim)
        # numpy.random.randn()
    
    def forward(self, input):
        self.input = input
        input = numpy.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant', constant_values = (0,0))        
        self.padded_input = input
        out = numpy.zeros(self.out_dim)
        for m in range(input.shape[0]):
            for filter_index in range(self.filter_count):
                for i in range(0, input.shape[1] - self.filter_dim[0] + 1, self.stride):
                    for j in range(0, input.shape[2] - self.filter_dim[1] + 1, self.stride):
                        subarr = input[m, i:i+self.filter_dim[0], j:j+self.filter_dim[1], :]
                        subarr = numpy.multiply(subarr, self.filters[:, :, :, filter_index])
                        out[m, i//self.stride, j//self.stride, filter_index] = \
                            numpy.sum(subarr) + float(self.bias[filter_index])
        return out

    def backward(self, dZ):
        db = numpy.zeros(self.bias.shape)
        dA_padded = numpy.zeros(self.padded_input.shape)
        dW = numpy.zeros(self.filters.shape)
        for m in range(self.padded_input.shape[0]):
            for filter_index in range(self.filter_count):
                for i in range(0, self.padded_input.shape[1] - self.filter_dim[0] + 1, self.stride):
                    for j in range(0, self.padded_input.shape[2] - self.filter_dim[1] + 1, self.stride):
                        subarr = self.padded_input[m, i:i+self.filter_dim[0], j:j+self.filter_dim[1], :]
                        dW[:, :, :, filter_index] += subarr*dZ[m, i//self.stride, j//self.stride, filter_index]
                        dA_padded[m, i:i+self.filter_dim[0], j:j+self.filter_dim[1], :] += self.filters[:,:,:,filter_index] * \
                            dZ[m, i//self.stride, j//self.stride, filter_index]
        for filter_index in range(self.filter_count):
            db[filter_index] = numpy.sum(dZ[:, :, :, filter_index])
        self.dA = dA_padded[:, self.padding:dA_padded.shape[1]-self.padding , self.padding:dA_padded.shape[2]-self.padding ,:]
        return self.dA


class Pooling:
    def __init__(self, dim, stride):
        self.dim = dim
        self.stride = stride
    
    def forward(self, input):
        self.input = input
        out = numpy.zeros((input.shape[0], (input.shape[1] - self.dim)//self.stride + 1,\
             (input.shape[2] - self.dim)//self.stride + 1, input.shape[3] ))
        for m in range(input.shape[0]):
            for filter_index in range(input.shape[3]):
                for i in range(0, input.shape[1] - self.dim + 1, self.stride):
                    for j in range(0, input.shape[2] - self.dim + 1, self.stride):
                        subarr = input[m, i:i+self.dim, j:j+self.dim, filter_index]
                        out[m, i//self.stride, j//self.stride, filter_index] = numpy.max(subarr)
        return out
    
    def backward(self, dZ):
        d_input = numpy.zeros(self.input.shape)
        for m in range(self.input.shape[0]):
            for filter_index in range(self.input.shape[3]):
                for i in range(0, self.input.shape[1] - self.dim + 1, self.stride):
                    for j in range(0, self.input.shape[2] - self.dim + 1, self.stride):
                        subarr = self.input[m, i:i+self.dim, j:j+self.dim, filter_index]
                        subMaxIndex = numpy.unravel_index(numpy.argmax(subarr), subarr.shape)
                        d_input[m, i+subMaxIndex[0], j + subMaxIndex[1], filter_index] += dZ[m, i//self.stride, j//self.stride, filter_index]
        return d_input



# numpy.random.seed(1)
# A_prev = numpy.random.randn(5, 5, 3, 2)
# hparameters = {"stride" : 1, "f": 2}
# dA = numpy.random.randn(5, 4, 2, 2)

# pool = Pooling(2, 1)
# A = pool.forward(A_prev)
# dA_prev = pool.backward(dA)
# print("mode = max")
# print('mean of dA = ', numpy.mean(dA))
# print('dA_prev[1,1] = ', dA_prev[1,1])  
# print()

class Activation:
    def __init__(self):
        a = 2
    def forward(self, input):
        out = numpy.where(input > 0, input, 0)
        self.input = input
        return out
    
    def backward(self, dZ):
        dA = numpy.where(self.input > 0, 1, 0)
        dA = numpy.multiply(dA, dZ) 
        return dA

# numpy.random.seed(1)
# a = numpy.random.randn(2, 3,3)
# print(a)
# active = Activation()
# b = active.forward(a)
# print(active.backward(a))

class FullyConnected:
    def __init__(self, output_dim, input_dim):
        self.out_dim = output_dim
        self.in_dim = input_dim
        input_size = 1
        for i in range(1, len(input_dim)):
            input_size = input_size*input_dim[i]

        self.bias = numpy.random.randn(*(output_dim, 1))
        self.W = numpy.random.randn(*(output_dim, input_size))

    def forward(self, input):
        input = input.reshape(input.shape[0], -1, 1)
        self.input = input
        out = numpy.zeros((input.shape[0], self.out_dim, 1))
        for m in range(input.shape[0]):
            out[m, :] = numpy.matmul(self.W, input[m])
        self.out = out
        return out

    def backward(self, dZ):
        dW = numpy.zeros(self.W.shape)
        db = numpy.zeros(self.bias.shape)
        d_input = numpy.zeros(self.input.shape)
        for m in range(self.input.shape[0]):
            dW += numpy.matmul(dZ[m], numpy.transpose(self.input[m]))
            db += dZ[m].sum(axis = 1).reshape(-1, 1)
            d_input[m] += numpy.matmul(numpy.transpose(self.W), dZ[m]) 
        d_input.reshape(self.in_dim)
        return d_input

# numpy.random.seed(1)
# a = numpy.random.randint(0,3, size=(2, 2, 3))
# print(a)
# c = numpy.random.randint(0,3, size=(2, 3))
# print(c.sum(axis = 0))
# fc = FullyConnected(4, a.shape)
# out = fc.forward(a)
# fc.backward(out)
# print(fc.forward(a))

class Softmax:
    def forward(self, input):
        out = numpy.zeros(input.shape)
        for m in range(input.shape[0]):
            out[m] = numpy.exp(input[m])
            out[m] = out[m]/numpy.sum(out[m])
        self.out = out
        return out

    def backward(self, y):
        return (self.out - y)/self.out.shape[0]

# a = numpy.array([[[1], [2]], [[3], [6]]])
# soft = Softmax()
# y = numpy.array([[[1], [0]], [[0], [1]]])
# print(soft.forward(a))
# print(soft.backward(y))

channel_count, row_count, column_count =  train_images[0].shape
numpy.random.seed(1)

conv = Convolution((6, 6, 1, 8), (32, 28, 28, 1), 2, 1)
pooling = Pooling(6, 2)
fc = FullyConnected(10, (32, 11, 11, 8))
soft = Softmax()

for x in range(0, 5):
    i = 0
    input = train_images[i:i+32]/255.0 - .5
    labels = train_labels[i:i+32].reshape(32, 1)
    out = conv.forward(input)
    out = pooling.forward(out)
    # print(out.shape)
    # print(out.shape, "=====")
    out = fc.forward(out)
    out = soft.forward(out)
    loss = 0
    yHat = out.reshape(out.shape[0], 10)
    for i in range(yHat.shape[0]):
        digit = numpy.argmax(yHat[i])
        if(digit != labels[i]):
            loss += 1
    print(loss)

    print(out.shape)

# print(channel_count, row_count, column_count)
# conv = Convolution((1, 2, 5), train_images[0].shape, 5, 1, 2)
# conv.forward(train_images[0])

# arr = [[[1, -2, 3], [4, 5, 6], [7, 8, 9]], [[11, -12, 13], [14, -15, 16], [17, 18, 19]]]
# arr = numpy.array(arr)
# print(arr)
# conv = FullyConnected(10, arr.shape)
# print(conv.forward(arr))


# print(numpy.max(arr))

# cv2.imshow("Image", train_images[5])
# print(train_labels[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()