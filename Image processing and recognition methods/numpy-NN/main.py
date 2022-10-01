from interface import *
import math
import numpy as np
# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "ReLU"

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, ...)), input values,
                    n - batch size, ... - arbitrary input shape
            :return: np.array((n, ...)), output values,
                    n - batch size, ... - arbitrary output shape (same as input)
        """
        # your code here \/
        return inputs * (inputs > 0)
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, ...)), dLoss/dOutputs,
                    n - batch size, ... - arbitrary output shape
            :return: np.array((n, ...)), dLoss/dInputs,
                    n - batch size, ... - arbitrary input shape (same as output)
        """
        # your code here \/
        inputs = self.forward_inputs
        return grad_outputs * (inputs > 0)
        # your code here /\


# ============================== 2.1.2 Softmax ===============================
class Softmax(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Softmax"

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, d)), input values,
                    n - batch size, d - number of units
            :return: np.array((n, d)), output values,
                    n - batch size, d - number of units
        """
        # your code here \/
        tmp = np.exp(inputs - np.max(inputs))
        return tmp / np.sum(tmp, axis=1)[:, None]
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                    n - batch size, d - number of units
            :return: np.array((n, d)), dLoss/dInputs,
                    n - batch size, d - number of units
        """
        # your code here \/
        outputs = self.forward_outputs
        return outputs * (grad_outputs - np.sum(grad_outputs * outputs, axis=1)[:, None])
        # your code here /\


# =============================== 2.1.3 Dense ================================
class Dense(Layer):

    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_shape = (units,)
        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None
        self.name = "Dense"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_units, = self.input_shape
        output_units, = self.output_shape

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter
        self.weights, self.weights_grad = self.add_parameter(
            name='weights',
            shape=(input_units, output_units),
            #initializer=normal_initializer()
            initializer=he_initializer(input_units)
        )

        self.biases, self.biases_grad = self.add_parameter(
            name='biases',
            shape=(output_units,),
            initializer=np.zeros
        )

    def num_of_parameters(self):
        return self.input_shape[0] * self.output_shape[0] + self.input_shape[0]
    def forward(self, inputs):
        """
            :param inputs: np.array((n, d)), input values,
                    n - batch size, d - number of input units
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of output units
        """
        batch_size, input_units = inputs.shape
        output_units, = self.output_shape
        return np.dot(inputs, self.weights) + self.biases
    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c)), dLoss/dOutputs,
                    n - batch size, c - number of output units
            :return: np.array((n, d)), dLoss/dInputs,
                    n - batch size, d - number of input units
        """
        # your code here \/
        batch_size, output_units = grad_outputs.shape
        input_units, = self.input_shape
        inputs = self.forward_inputs

        # Don't forget to update current gradients:
        # dLoss/dWeights
        self.weights_grad[...] = np.dot(inputs.T, grad_outputs) / batch_size
        # dLoss/dBiases
        self.biases_grad[...] = np.mean(grad_outputs, axis=0)
        return np.dot(grad_outputs, self.weights.T)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def __call__(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values
            :return: np.array((n,)), loss scalars for batch
        """
        # your code here \/
        batch_size, output_units = y_gt.shape
        return np.sum(-y_gt * np.log(y_pred), axis=1)
        # your code here /\

    def gradient(self, y_gt, y_pred):
        """
            :param y_gt: np.array((n, d)), ground truth (correct) labels
            :param y_pred: np.array((n, d)), estimated target values
            :return: np.array((n, d)), gradient loss to y_pred
        """
        # your code here \/
        return -y_gt / (y_pred)
        # your code here /\


# ================================ 2.3.1 SGD =================================
class SGD(Optimizer):
    def __init__(self, lr):
        self._lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter
            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam
                :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            return parameter - self._lr * parameter_grad
            # your code here /\

        return updater


# ============================ 2.3.2 SGDMomentum =============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self._lr = lr
        self._momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
            :param parameter_shape: tuple, the shape of the associated parameter
            :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
                :param parameter: np.array, current parameter values
                :param parameter_grad: np.array, current gradient, dLoss/dParam
                :return: np.array, new parameter values
            """
            # your code here \/
            assert parameter_shape == parameter.shape
            assert parameter_shape == parameter_grad.shape
            assert parameter_shape == updater.inertia.shape

            updater.inertia[...] = parameter - self._lr * parameter_grad
            return self._momentum + updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ======================= 2.4 Train and test on MNIST ========================
def train_mnist_model(x_train, y_train, x_valid, y_valid):

    # your code here \/
    # 1) Create a Model
    model = Model(CategoricalCrossentropy(), SGD(0.05))

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(units=80, input_shape=(784,)))
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())
    model.print_parameters()
    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, batch_size=10, epochs=2, x_valid=x_valid, y_valid=y_valid)
    # your code here /\
    return model

# ------------------------ Conv2D 3.1 ----------------------------------

"""
    Method which calculates the padding based on the specified output shape and the
    shape of the filters
"""
def determine_padding(filter_shape, output_shape):
    # No padding
    if output_shape == "valid":
        return (0, 0), (0, 0)
    # Pad so that the output shape is the same as input shape (given that stride=1)
    elif output_shape == "same":
        filter_height, filter_width = filter_shape

        # Derived from:
        # output_height = (height + pad_h - filter_height) / stride + 1
        # In this case output_height = height and stride = 1. This gives the
        # expression for the padding below.
        pad_h1 = int(math.floor((filter_height - 1) / 2))
        pad_h2 = int(math.ceil((filter_height - 1) / 2))
        pad_w1 = int(math.floor((filter_width - 1) / 2))
        pad_w2 = int(math.ceil((filter_width - 1) / 2))

        return (pad_h1, pad_h2), (pad_w1, pad_w2)

def get_im2col_indices(images_shape, filter_shape, padding, stride):
    """
        :param images_shape: (n, c, h, w), images shape,
                n - batch size, c - number of input channels
                (h, w) - input image shape
        :param filter_shape: (fh, fw),
        :param padding: (int, int),
                Specifies how much to pad edges
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :return: np.array((n, oc, oh, ow)), output values,
                n - batch size, oc - number of output channels
                (oh, ow) - output image shape
    """
    batch_size, channels, height, width = images_shape
    filter_height, filter_width = filter_shape
    pad_h, pad_w = padding
    out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
    out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)
    # your code here \/
    i = channels * filter_shape[0] * filter_shape[1]
    j = ((width + pad_w[0] + pad_w[1] - filter_shape[1]) // stride + 1) * ((height + pad_h[0] + pad_h[1] - filter_shape[0]) // stride + 1)
    k = np.zeros((i, 1), dtype='int64')
    off = i // channels
    for m in range(channels):
        k[off:] += 1
        off += i // channels
    new_i = np.zeros((i, j), dtype='int64')
    new_j = np.zeros((i, j), dtype='int64')
    cnt = 0
    for m in range(i):
        s = 0
        for t in range(0, j, j // out_height):
            for l in range(j // out_height):
                if t + l >= j:
                    break
                new_i[m][t + l] = cnt // filter_shape[0] + s * stride
            s += 1
        cnt = (cnt + 1) % (filter_shape[0] * filter_shape[1])
    cnt = 0
    for m in range(i):
        for t in range(0, j, out_width):
            for l in range(out_width):
                if t + l >= j:
                    break
                new_j[m][t + l] = cnt + l * stride
        cnt = (cnt + 1) % filter_shape[0]
    cnt = 0
    return k, new_i, new_j
    # your code here /\

"""
    Method which turns the image shaped input to column shape.
    Used during the forward pass.
"""
def im2col(images, filter_shape, stride, output_shape):
    """
        :param images: np.ndarray((n, c, h, w)), images ,
                n - batch size, c - number of input channels
                (h, w) - input image shape
        :param filter_shape: (fh, fw)
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :param output_shape: "same" or "valid"
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
        :return: np.array((oc, os)), image in column shape
                oc - fh * fw * c, os - output size
    """
    filter_height, filter_width = filter_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    # Add padding to the image
    images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode = 'constant')
    # Calculate the indices where the dot products are to be applied between weights
    # and the image
    k0, i0, j0 = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)
    # your code here \/
    
    batch_size = images_padded.shape[0]
    cols = images_padded[:, k0, i0, j0]
    cols = cols.transpose(1, 2, 0)
    cols = cols.reshape(filter_height * filter_width * images_padded.shape[1], -1)
    return cols
    # your code here /\
"""
    Method which turns the column shaped input to image shape.
    Used during the backward pass.
"""
def col2im(cols, images_shape, filter_shape, stride, output_shape):
    """
        :param cols: np.ndarray((ic, is)), images in column shape
        :param filter_shape: (fh, fw)
        :param stride: int
                Specifies how far the convolution window moves for each step.
        :param output_shape: "same" or "valid"
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
        :return: np.ndarray((n, c, h, w)), images ,
                n - batch size, c - number of input channels
                (h, w) - input image shape
    """
    batch_size, channels, height, width = images_shape
    pad_h, pad_w = determine_padding(filter_shape, output_shape)
    height_padded = height + np.sum(pad_h)
    width_padded = width + np.sum(pad_w)
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded), dtype = cols.dtype)

    # Calculate the indices where the dot products are applied between weights
    # and the image
    k0, i0, j0 = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

    # your code here \/
    images = cols.reshape(channels * filter_shape[0] * filter_shape[1], -1, batch_size)
    images = images.transpose(2, 0, 1)
    np.add.at(images_padded, (slice(None), k0, i0, j0), images)
    if pad_h[0] == 0 and pad_h[1] == 0 and pad_w[0] == 0 and pad_w[1] == 0:
        return images_padded
    return images_padded[:,:, pad_h[0]:-pad_h[1], pad_w[0]:-pad_w[1]]
    # Return image without padding
    # your code here /\


class Conv2D(Layer):
    def __init__(self, filters, kernel_size = 3, padding = "same", strides = 1, *args, **kwargs):
        """
            filters: int
                Number of filters, number of channels of the output shape
            kernel_size: int
                Kernel size
            padding: "valid" or "same". 
                "same" - add padding so that the output height and width matches the
                input height and width. "valid" - no padding is added
            strides: int
                Specifies how far the convolution window moves for each step.
        """
        super().__init__(*args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.name = "Conv2D"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter
        self.filter_shape = (self.kernel_size, self.kernel_size)

        self.output_shape = self.get_output_shape()

        self.weights, self.weights_grad = self.add_parameter(
            name = "weights",
            shape = (self.filters, self.input_shape[0], *self.filter_shape),
            initializer = normal_initializer()
        )

        self.biases, self.biases_grad = self.add_parameter(
            name = 'biases',
            shape = (self.filters, 1),
            initializer = np.zeros
        )

    # returns the number of parameters
    def num_of_parameters(self):
        return self.kernel_size * self.kernel_size * self.filters + self.filters

    # returns shape of the output tensor
    def get_output_shape(self):
        channels, height, width = self.input_shape
        pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
        output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.strides + 1
        output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.strides + 1
        return self.filters, int(output_height), int(output_width)

    def forward(self, inputs):
        """
            :param inputs: np.array((n, ic, ih, iw)), input values,
                    n - batch size, ic - number of input channels
                    (ih, iw) - input image shape
            :return: np.array((n, oc, oh, ow)), output values,
                    n - batch size, oc - number of output channels
                    (oh, ow) - output image shape
        """
        batch_size = inputs.shape[0]
        # your code here \/
        weights = self.weights.reshape(self.filters, -1)

        self.cols = im2col(inputs, self.filter_shape, self.strides, self.padding)
        self.basic_shape = inputs.shape

        outputs = np.matmul(weights, self.cols) + self.biases
        outputs = outputs.reshape(self.output_shape + (batch_size,))
        outputs = outputs.transpose(3, 0, 1, 2)

        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, oc, oh, ow)), dLoss/dOutputs,
                    n - batch size, oc - number of output filters
                    (oh, ow) - output image shape
            :return: np.array((n, ic, ih, iw)), dLoss/dInputs,
                    n - batch size, ic - number of input filters
                    (ih, iw) - input image shape
        """
        # Reshape accumulated gradient into column shape
        #batch_size = inputs.shape[0]
        # your code here \/

        bias_grad = grad_outputs.transpose(1, 2, 3, 0)
        self.biases_grad = np.sum(bias_grad, axis=(1, 2, 3)).reshape(self.filters, -1)


        gr_col = grad_outputs.transpose(1, 2, 3, 0).reshape(self.filters, -1)
        self.weights_grad = np.matmul(gr_col, self.cols.T)
        self.weights_grad = self.weights_grad.reshape(self.weights.shape)

        weights = self.weights.reshape(self.filters, -1)
        grad_x_col = np.matmul(weights.T, gr_col)
        grad_inputs = col2im(grad_x_col, self.basic_shape, self.filter_shape, self.strides, self.padding)

        return grad_inputs
        # your code here /\

# ======================= 3.2 MaxPooling Layer ==========================

class MaxPooling2D(Layer):
    def __init__(self, pool_size = 2, strides = 1, *args, **kwargs):
        """
            pool_size: int
                Size of the pooling window
            strides: int
                Specifies how far the pooling window moves for each step.
        """
        super().__init__(*args, **kwargs)
        self.pool_size = (pool_size, pool_size)
        self.strides = strides
        self.name = "MaxPool2D"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = self.get_output_shape()

    def num_of_parameters(self):
        return 0

    # returns shape of the output tensor
    def get_output_shape(self):
        channels, height, width = self.input_shape
        out_height = (height - self.pool_size[0]) / self.strides + 1
        out_width = (width - self.pool_size[1]) / self.strides + 1
        assert out_height % 1 == 0
        assert out_width % 1 == 0
        return channels, int(out_height), int(out_width)

    def check_max(self, num, b, c, i, j, max_i, max_j):
        if self.max < num:
            self.max = num
            self.max_indices[b][c][i][j][0] = max_i
            self.max_indices[b][c][i][j][1] = max_j

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, ih, iw)), input values,
                    n - batch size, c - number of input channels
                    (ih, iw) - input image shape
            :return: np.array((n, c, oh, ow)), output values,
                    n - batch size, c - number of input channels
                    (oh, ow) - output image shape
        """
        batch_size, channels, height, width = inputs.shape
        _, out_height, out_width = self.output_shape
        # your code here \/
        outputs = np.empty((batch_size, ) + self.output_shape)
        self.max_indices = np.zeros(outputs.shape + (2,), dtype='int')
        for b in range(batch_size):
            for c in range(self.output_shape[0]):
                for i in range(0, self.output_shape[1]):
                    for j in range(0,self.output_shape[2]):
                        self.max = inputs[b][c][2 * i][2 * j]
                        self.max_indices[b][c][i][j][0] = 2 * i
                        self.max_indices[b][c][i][j][1] = 2 * j
                        self.check_max(inputs[b][c][2 * i + 1][2 * j], b, c, i, j, 2 * i + 1, 2 * j)
                        self.check_max(inputs[b][c][2 * i + 1][2 * j + 1], b, c, i, j, 2 * i + 1, 2 * j + 1)
                        self.check_max(inputs[b][c][2 * i][2 * j + 1], b, c, i, j, 2 * i, 2 * j + 1)
                        outputs[b][c][i][j] = self.max
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, oh, ow)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (oh, ow) - output image shape
            :return: np.array((n, c, ih, iw)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (ih, iw) - input image shape
        """
        batch_size, _, _, _ = grad_outputs.shape
        channels, height, width = self.input_shape
        # your code here \/
        
        grad_inputs = np.zeros((batch_size, ) + self.input_shape)
        shape = self.max_indices.shape
        for b in range(shape[0]):
            for c in range(shape[1]):
                for i in range(shape[2]):
                    for j in range(shape[3]):
                        grad_inputs[b][c][self.max_indices[b][c][i][j][0]][self.max_indices[b][c][i][j][1]] = grad_outputs[b][c][i][j]
        return grad_inputs
        # your code here /\

# ============================ 3.3 Global Average Pooling ========================

class GlobalAveragePooling(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        c, h, w = self.input_shape
        self.output_shape = (c,)

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, ih, iw)), input values,
                    n - batch size, c - number of channels
                    (ih, iw) - input image shape
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of channels
        """
        # your code here \/
        batch_size, channels, height, width = inputs.shape
        outputs = np.mean(inputs, axis=(2, 3))
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, d)), dLoss/dOutputs,
                    n - batch size, d - number of channels
            :return: np.array((n, ih, iw, d)), dLoss/dInputs,
                    n - batch size, d - number of channels
                    (ih, iw) - input image shape
        """
        batch_size = grad_outputs.shape[0]
        # your code here \/
        grad_inputs = np.ones((batch_size, ) + self.input_shape) / self.input_shape[1] / self.input_shape[2]
        return grad_inputs * grad_outputs[:,:,None, None]
        # your code here /\

# ============================ 3.4 Batch Normalization ========================

class BatchNormalization(Layer):
    def __init__(self, momentum = 0.99, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "BatchNorm"
        self.momentum = momentum
        self.eps = 0.01
        self.running_mean = None
        self.running_var = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.gamma, self.gamma_grad = self.add_parameter(
            name = "gamma",
            shape = (self.input_shape),
            initializer = np.ones
        )

        self.beta, self.beta_grad = self.add_parameter(
            name = 'beta',
            shape = (self.input_shape),
            initializer = np.zeros
        )

    def num_of_parameters(self):
        return self.gamma.size + self.beta.size

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        self.inputs = inputs
        # your code here \/
        self.cnt = 0
        if self.running_mean is None:
            self.running_mean = np.zeros((inputs.shape[1:]))
            self.running_var = np.zeros((inputs.shape[1:]))
        if self.training:
            
            self.means = np.mean(inputs, axis=0)
            self.var = np.var(inputs, axis=0)
            self.mean_input = (inputs - self.means) / np.sqrt(self.var + self.eps)
            outputs = self.gamma * self.mean_input + self.beta

            self.running_mean = (self.cnt * self.running_mean +  self.means) / (self.cnt + 1)
            self.running_var = (self.cnt * self.running_var + self.var) / (self.cnt + 1)

            self.cnt += 1

        else:
            outputs = (inputs - self.running_mean) / np.sqrt(self.running_var + self.eps)
            outputs= outputs * self.gamma + self.beta

        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = grad_outputs.shape[0]

        # your code here \/
        self.gamma_grad = np.sum(grad_outputs * self.mean_input, axis=0)
        self.beta_grad = np.sum(grad_outputs, axis=0)
        
        disp = 1 / np.sqrt(self.var + self.eps)
        mean_input_grad = grad_outputs * self.gamma
        disp_grad = -np.sum(mean_input_grad * (self.inputs - self.means), axis=0) / 2 * (disp ** 3)
        mean_grad = -np.sum(mean_input_grad * disp, axis=0) - disp_grad * np.sum(2 * (self.inputs - self.means), axis=0) / batch_size
        grad_inputs = mean_input_grad * disp + 2 * disp_grad * (self.inputs - self.means) / batch_size  + mean_grad / batch_size
        return grad_inputs
        # your code here /\


# ============================= 3.5 Dropout ===============================

class Dropout(Layer):
    def __init__(self, p = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Dropout"
        self.p = p

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        # When testing on unittests, please uncomment next two lines
        np.random.seed(1)
        self.training = True

    def num_of_parameters(self):
        return 0

    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            for conv layers
            or
            :param inputs: np.array((n, c)), input values,
                    n - batch size, c - number of units
            :return: np.array((n, c)), output values,
                    n - batch size, c - number of units
            for dense layers
        """
        # your code here \/
        self.dropout = np.zeros(inputs.shape)
        for i in range(inputs.shape[0]):
            elems = 1
            for size in inputs.shape[1:]:
                elems *= size
            self.dropout[i] = (np.random.rand(elems) > self.p).astype(int).reshape(inputs.shape[1:])
        return inputs * self.dropout
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs * self.dropout
        # your code here /\

# ============================ 3.6 Flatten Layer ============================

class Flatten(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "Flatten"

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.output_shape = (np.array(self.input_shape[:]).prod(), )

    def num_of_parameters(self):
        return 0
    def forward(self, inputs):
        """
            :param inputs: np.array((n, c, h, w)), input values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, (c * h * w))), output values,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        # your code here \/
        outputs = inputs.reshape(-1, np.prod(inputs.shape[1:]))
        return outputs
        # your code here /\

    def backward(self, grad_outputs):
        """
            :param grad_outputs: np.array((n, (c * h * w))), dLoss/dOutputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
            :return: np.array((n, c, h, w)), dLoss/dInputs,
                    n - batch size, c - number of channels
                    (h, w) - input image shape
        """
        batch_size = grad_outputs.shape[0]
        # your code here \/
        return grad_outputs.reshape((batch_size, ) + self.input_shape)
        # your code here /\

# ======================= 3.7 Train Cifar10 Conv Model ========================

def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(
        CategoricalCrossentropy(), SGD(0.05)
    )
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    shape = x_train[0].shape
    model.add(Flatten(input_shape=x_train[0].shape))
    model.add(Dense(units=500))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.15))
    model.add(Dense(units=350))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dropout(0.1))
    model.add(Dense(units=150))
    model.add(BatchNormalization())
    model.add(ReLU())
    model.add(Dense(units=10))
    model.add(Softmax())
    model.print_parameters()

    # 3) Train and validate the model using the provided data
    model.fit(x_train=x_train, y_train=y_train, batch_size=60, epochs=25, x_valid=x_valid, y_valid=y_valid)

    # your code here /\
    return model