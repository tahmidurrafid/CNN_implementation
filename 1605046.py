from copy import deepcopy
from multiprocessing import Pool, pool
import idx2numpy
import numpy
import cv2

class CNN:
    learning_rate = 0.1
    batch_size = 32

class Convolution:
    def __init__(self, filter_dim, in_dim, padding, stride):
        self.padding = padding
        self.stride = stride
        self.filter_count = filter_dim[3]
        self.filter_dim = filter_dim
        self.in_dim = in_dim
        self.input_changed_dim = in_dim
        self.out_dim = (in_dim[0] , (in_dim[1] + 2*self.padding - self.filter_dim[0])//self.stride + 1, \
            (in_dim[2] + 2*self.padding - self.filter_dim[1])//self.stride  + 1, self.filter_count)
        # self.bias = numpy.random.randn(self.filter_count)
        self.bias = numpy.zeros(self.filter_count)
        self.filters = numpy.random.randn(*filter_dim)*0.001
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
        self.filters -= dW*CNN.learning_rate
        self.bias -= db*CNN.learning_rate
        return self.dA


class Pooling:
    def __init__(self, in_dim, dim, stride):
        self.dim = dim
        self.stride = stride
        self.out_dim = (in_dim[0], (in_dim[1] - self.dim)//self.stride + 1,\
             (in_dim[2] - self.dim)//self.stride + 1, in_dim[3] )
    
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
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = self.in_dim

    def forward(self, input):
        out = numpy.where(input > 0, input, input)
        self.input = input
        return out
    
    def backward(self, dZ):
        dA = numpy.where(self.input > 0, 1, 0)
        dA = numpy.multiply(dA, dZ) 
        return dZ
    # def forward(self, input):
    #     out = numpy.where(input > 0, input, 0)
    #     self.input = input
    #     return out
    
    # def backward(self, dZ):
    #     dA = numpy.where(self.input > 0, 1, 0)
    #     dA = numpy.multiply(dA, dZ) 
    #     return dA


class FullyConnected:
    def __init__(self, output_dim, input_dim):
        self.out_channel = output_dim
        self.in_dim = input_dim
        input_size = 1
        for i in range(1, len(input_dim)):
            input_size = input_size*input_dim[i]
        self.out_dim = (input_dim[0], self.out_channel, 1)
        # self.bias = numpy.random.randn(*(output_dim, 1))
        self.bias = numpy.zeros((output_dim, 1))
        self.W = numpy.random.randn(*(output_dim, input_size))*0.001

    def forward(self, input):
        self.original_in_dim = input.shape
        input = input.reshape(input.shape[0], -1, 1)
        self.input = input
        out = numpy.zeros((input.shape[0], self.out_channel, 1))
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
        d_input = d_input.reshape(self.original_in_dim)
        self.bias -=  db*CNN.learning_rate
        self.W -= dW*CNN.learning_rate
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
    def __init__(self, in_dim):
        self.in_dim = in_dim
        self.out_dim = in_dim

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


def mnistData():
    train_images = 'mnist/train-images.idx3-ubyte'
    train_labels = 'mnist/train-labels.idx1-ubyte'
    train_images = idx2numpy.convert_from_file(train_images)
    train_labels = idx2numpy.convert_from_file(train_labels)
    train_images = train_images.reshape(-1, 28, 28, 1)

    test_images = 'mnist/t10k-images.idx3-ubyte'
    test_labels = 'mnist/t10k-labels.idx1-ubyte'
    test_images = idx2numpy.convert_from_file(test_images)
    test_labels = idx2numpy.convert_from_file(test_labels)
    test_images = test_images.reshape(-1, 28, 28, 1)
    numpy.random.seed(1)

    conv = Convolution((6, 6, 1, 5), (CNN.batch_size, 28, 28, 1), 1, 2)
    relu = Activation(conv.out_dim)
    pooling = Pooling(conv.out_dim, 2, 2)
    fc = FullyConnected(10, pooling.out_dim)
    soft = Softmax(fc.out_dim)

    def run(img, lab, limit):
        table = numpy.zeros((10, 10))
        ceL = 0
        for j in range(0, limit, CNN.batch_size):
            print(j, ": ")
            input = img[j:j+CNN.batch_size]/255.0 - .5
            labels = lab[j:j+CNN.batch_size].reshape(CNN.batch_size, 1)

            out = conv.forward(input)
            out = relu.forward(out)
            out = pooling.forward(out)
            out = fc.forward(out)
            out = soft.forward(out)
            loss = 0
            yHat = out.reshape(out.shape[0], 10)
            proc_labels = numpy.zeros((labels.shape[0], 10, 1))
            for i in range(yHat.shape[0]):
                digit = numpy.argmax(yHat[i])
                actual = labels[i][0]
                table[actual][digit] += 1
                ceL += actual
                if(digit != labels[i][0]):
                    loss += 1
                proc_labels[i, int(labels[i,0])] = 1
                ceL += numpy.log(yHat[i][actual])
                # print(yHat[i][actual], numpy.log(yHat[i][actual]))
            print(loss)
            print(out.shape)

            dout = soft.backward(proc_labels)
            dout = fc.backward(dout)
            dout = pooling.backward(dout)
            dout = relu.backward(dout)
            dout = conv.backward(dout)
        print(table)
        tp = numpy.trace(table)
        precision = numpy.zeros(10)
        recall = numpy.zeros(10)
        f1 = numpy.zeros(10)
        for i in range(0, 10):
            precision[i] = table[i,i]/numpy.sum(table[:,i])
            recall[i] = table[i,i]/numpy.sum(table[i,:])
            f1[i] = 2/(1.0/precision[i] + 1.0/recall[i])
        accuracy = tp/numpy.sum(table)
        macroF1 = numpy.mean(f1)
        print("Accuracy: ", accuracy)
        print("macro-f1: ", macroF1)
        print("Loss: ", ceL/numpy.sum(table))

    run(train_images, train_labels, 1000)
    run(test_images, test_labels, 1000)
    # print(test_images.shape)

mnistData()

# conv = Convolution((5, 5, 1, 6), (CNN.batch_size, 28, 28, 1), 2, 1)
# relu = Activation(conv.out_dim)
# pooling = Pooling(relu.out_dim, 2, 2)
# conv2 = Convolution((5, 5, pooling.out_dim[3], 12), pooling.out_dim, 0, 1)
# relu2 = Activation(conv2.out_dim)
# pooling2 = Pooling(relu2.out_dim, 2, 2)
# conv3 = Convolution((5, 5, pooling2.out_dim[3], 50), pooling2.out_dim, 0, 1)
# relu3 = Activation(conv3.out_dim)
# fc = FullyConnected(10, relu3.out_dim)
# soft = Softmax(fc.out_dim)

# for x in range(0, 1000):
#     j = x*CNN.batch_size
#     input = train_images[j:j+CNN.batch_size]/255.0 - .5
#     labels = train_labels[j:j+CNN.batch_size].reshape(CNN.batch_size, 1)

#     out = conv.forward(input)
#     out = relu.forward(out)
#     out = pooling.forward(out)
#     out = conv2.forward(out)
#     out = relu2.forward(out)
#     out = pooling2.forward(out)
#     out = conv3.forward(out)
#     out = relu3.forward(out)
#     out = fc.forward(out)
#     out = soft.forward(out)

#     loss = 0
#     yHat = out.reshape(out.shape[0], 10)
#     proc_labels = numpy.zeros((labels.shape[0], 10, 1))
#     for i in range(yHat.shape[0]):
#         digit = numpy.argmax(yHat[i])
#         if(digit != labels[i][0]):
#             loss += 1
#         proc_labels[i, int(labels[i,0])] = 1
#     print(loss)
#     print(out.shape)
#     dout = soft.backward(proc_labels)
#     dout = fc.backward(dout)
#     dout = relu3.backward(dout)
#     dout = conv3.backward(dout)
#     dout = pooling2.backward(dout)
#     dout = relu2.backward(dout)
#     dout = conv2.backward(dout)
#     dout = pooling.backward(dout)
#     dout = relu.backward(dout)
#     dout = conv.backward(dout)


# print(numpy.max(arr))

# cv2.imshow("Image", train_images[5])
# print(train_labels[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()