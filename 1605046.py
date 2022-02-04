from copy import deepcopy
import idx2numpy
import numpy
import cv2

train_images = 'mnist/train-images.idx3-ubyte'
train_labels = 'mnist/train-labels.idx1-ubyte'
train_images = idx2numpy.convert_from_file(train_images)
train_labels = idx2numpy.convert_from_file(train_labels)
train_images = numpy.array([x.reshape(1, 28, 28) for x in train_images])

class Convolution:
    def __init__(self, filter_dim, dim, filter_count, padding, stride):
        self.padding = padding
        self.stride = stride
        self.filter_count = filter_count
        self.filter_dim = filter_dim
        self.out_dim = ((dim[1] + 2*self.padding - self.filter_dim[1])//self.stride + 1, \
            (dim[2] + 2*self.padding - self.filter_dim[2])//self.stride  + 1)
        
        self.bias = numpy.random.randint(0, 255, size = filter_dim)
        self.filters = numpy.random.randint(0, 255, \
            size = (self.filter_count , filter_dim[0],filter_dim[1], filter_dim[2]))
    
    def forward(self, input):
        padded = []
        for i in range(len(input)):
            padded.append(numpy.pad(input[i], self.padding, mode='constant'))            
        input = numpy.array(padded)
        out = []
        for filter_index in range(self.filter_count):
            filtered = []
            for i in range(0, input.shape[1] - self.filter_dim[1] + 1, self.stride):
                row = []
                for j in range(0, input.shape[2] - self.filter_dim[2] + 1, self.stride):
                    subarr = input[0:self.filter_dim[0], i:i+self.filter_dim[1], j:j+self.filter_dim[2]]
                    subarr = subarr*self.filters[filter_index] + self.bias
                    row.append(numpy.sum(subarr))
                filtered.append(row)
            out.append(filtered)
        out = numpy.array(out)
        return out

    def print(self):
        print(self.filters)

class Pooling:
    def __init__(self, dim, stride):
        self.dim = dim
        self.stride = stride
    
    def forward(self, input):
        out = []
        for filter_index in range(input.shape[0]):
            filtered = []
            for i in range(0, input.shape[1], self.stride):
                row = []
                for j in range(0, input.shape[2], self.stride):
                    subarr = input[filter_index, i:min(i+self.dim, input.shape[1]), j:min(j+self.dim, input.shape[2])]
                    row.append(numpy.max(subarr))
                filtered.append(row)
            out.append(filtered)
        out = numpy.array(out)
        return out        

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
        return out

channel_count, row_count, column_count =  train_images[0].shape
# print(channel_count, row_count, column_count)

conv = Convolution((1, 2, 5), train_images[0].shape, 5, 1, 2)
conv.forward(train_images[0])



arr = [[[1, -2, 3], [4, 5, 6], [7, 8, 9]], [[11, -12, 13], [14, -15, 16], [17, 18, 19]]]
arr = numpy.array(arr)
print(arr)
conv = Activation()
print(arr.reshape(1, -1))

# print(numpy.max(arr))

# cv2.imshow("Image", train_images[5])
# print(train_labels[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()