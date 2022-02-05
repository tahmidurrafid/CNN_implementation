from copy import deepcopy
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
        self.bias = numpy.random.randint(0, 255, size = self.filter_count)
        self.filters = numpy.random.randint(0, 255, \
            size = filter_dim)
    
    def forward(self, input):
        self.input = input
        input = numpy.pad(input, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant', constant_values = (0,0))        
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


class Pooling:
    def __init__(self, dim, stride):
        self.dim = dim
        self.stride = stride
    
    def forward(self, input):
        out = numpy.zeros((input.shape[0], (input.shape[1] - self.dim)//self.stride + 1,\
             (input.shape[2] - self.dim)//self.stride + 1, input.shape[3] ))
        for m in range(input.shape[0]):
            for filter_index in range(input.shape[3]):
                for i in range(0, input.shape[1] - self.dim + 1, self.stride):
                    for j in range(0, input.shape[2] - self.dim + 1, self.stride):
                        subarr = input[m, i:i+self.dim, j:j+self.dim, filter_index]
                        out[m, i//self.stride, j//self.stride, filter_index] = numpy.max(subarr)
                        # row.append(numpy.max(subarr))
        return out        

numpy.random.seed(1)
A_prev = numpy.random.randn(2, 5, 5, 3)
# hparameters = {"stride" : 1, "f": 3}
pool = Pooling(3, 1)
A = pool.forward(A_prev)
print("mode = max")
print("A.shape = " + str(A.shape))
print("A =\n", A)
print()

class Activation:
    def __init__(self):
        a = 2
    def forward(self, input):
        out = numpy.where(input > 0, input, 0)
        return out

class FullyConnected:
    def __init__(self, output_dim, input_dim):
        self.out_dim = output_dim
        self.in_dim = input_dim
        input_size = input_dim[0]*input_dim[1]*input_dim[2]
        self.bias = numpy.random.randint(0, 255, size = (output_dim, 1))
        self.W = numpy.random.randint(0, 255, \
            size = (output_dim, input_size))

    def forward(self, input):
        input = input.reshape(-1, 1)
        out = numpy.matmul(self.W, input) + self.bias
        return out.reshape(-1)

class Softmax:
    def forward(self, input):
        out = numpy.exp(input)
        out = out/numpy.sum(out)
        return out

channel_count, row_count, column_count =  train_images[0].shape


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