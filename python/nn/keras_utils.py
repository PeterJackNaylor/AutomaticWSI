
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops

from keras.layers import Activation, Concatenate, GaussianDropout
from keras.layers import Conv1D, GlobalAveragePooling1D, GlobalMaxPooling2D
from keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling1D
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import Flatten, multiply, Lambda

from keras.models import Sequential, Model
from keras.engine import Layer, InputSpec

#from keras.models import Sequential
from keras import regularizers

def conv_shape_bn_act_drop(i_layer, hidden_layer, weight_decay, 
                           drop_out_rate, input_size, name="bottleneck_1",
                           activation='relu'):
    """Convolutional layer followed by BatchNorm, activation and DropOut : to use when defining a first layer.
    
    Parameters
    ----------
    i_layer : Tensor
        Input layer
    hidden_layer : int
        number of filters 
    weight_decay : float
        weight decay rate
    drop_out_rate : float
        drop out rate
    input_size : tuple
        size of the input layer, 
    name : str, optional
        name of the layer, by default "bottleneck_1"
    activation : str, optional
        name of the activation function, by default 'relu'
    
    Returns
    -------
    Tensor
        Output Tensor.
    """
    x_i = Conv1D(hidden_layer, 1, strides=1, activation=None,
                 use_bias=True, kernel_initializer="glorot_normal",
                 bias_initializer="glorot_uniform", 
                 kernel_regularizer=regularizers.l2(weight_decay),
                 name=name, input_shape=input_size)(i_layer)
    x_i = BatchNormalization()(x_i)
    if activation is not None:
        x_i = Activation(activation)(x_i)
    if drop_out_rate != 0:
        x_i = Dropout(drop_out_rate)(x_i)
    return x_i

def dense_bn_act_drop(i_layer, number_of_filters, name, wd, dr):
    """Dense layer followed by BatchNorm, ReLU activation and Dropout.
    
    Parameters
    ----------
    i_layer : Tensor
        Input Tensor
    number_of_filters : int
        Dimension of the output.
    name : str
        name of the dense layer
    wd : float
        weight decay.
    dr : float
        dropout rate (between 0 and 1)
    Returns
    -------
    Tensor
        Output Tensor.
    """

    x_i = Dense(number_of_filters, activation=None, use_bias=True,
                kernel_initializer="glorot_normal",
                bias_initializer="glorot_uniform",
                kernel_regularizer=regularizers.l2(wd),
                name=name)(i_layer)
    x_i = BatchNormalization()(x_i)
    x_i = Activation('relu')(x_i)
#    x_i = Concatenate(axis=-1)([i_layer, x_i])
    x_i = Dropout(dr)(x_i)
    return x_i

class KMaxPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension).
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        
        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)
        top_k_v2 = tf.batch_gather(shifted_input, top_k_indices)
        # return flattened output
        return Flatten()(top_k_v2)

class KMinPooling(Layer):
    """
    K-min pooling layer that extracts the k-lowest activations from a sequence (2nd dimension).
    TensorFlow backend.
    
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        
        # swap last two dimensions since top_k will be applied along the last dimension
        shifted_input = tf.transpose(inputs, [0, 2, 1])
        neg_shifted_input = tf.scalar_mul(tf.constant(-1, dtype="float32"), shifted_input)

        # extract top_k, returns two tensors [values, indices]
        top_k = tf.nn.top_k(neg_shifted_input, k=self.k, sorted=True, name=None)[0]
        
        # return flattened output
        return Flatten()(top_k)

def WeldonPooling(x_i, k):
    """Performs a Weldon Pooling, that is: selects the K-highest and the K-lowest activations. 
    
    Parameters
    ----------
    x_i : list
        list of  activation to pool.
    k : int
       number of highest and lowest statistics to pool. 
    
    Returns
    -------
    list
        Pulled summary statistics of length 2*R       
    """
    max_x_i = KMaxPooling(k=k)(x_i)

    neg_x_i = KMinPooling(k=k)(x_i)

    x_i = Concatenate(axis=-1)([max_x_i, neg_x_i])

    return x_i

def fetch_top_k_in_x(params, y_ind, k):
    bs = tf.shape(y_ind)[0]
    tt = tf.reshape(tf.stack([gen_math_ops._range(0, bs, 1) for _ in range(k)], axis=-1), [-1])
    y_ind = tf.reshape(y_ind, [-1])
    indice = tf.stack([tt, y_ind], axis=-1)
    indice =  tf.reshape(indice, [bs, k, -1])
    return tf.gather_nd(params, indice)

class KMaxConcPooling(Layer):
    """
    K-max pooling layer that extracts the k-highest activations from a sequence (2nd dimension)
    and concatenates the previous bottleneck representation to this score.
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        #self.input_spec = InputSpec(dtype=list, ndim=3)
        self.k = k
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        a, b = input_shape
        return (b[0], (a[2] * self.k), b[-1] + 1)

    def call(self, inputs):
        assert isinstance(inputs, list)
        s, x = inputs                
        shifted_s = tf.transpose(s, [0, 2, 1])

        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(shifted_s, k=self.k, sorted=True, name=None)
        s_top_k = tf.transpose(top_k, [0, 2, 1])
        x_top_k = fetch_top_k_in_x(x, top_k_indices, self.k)
        x_top_k = tf.concat([s_top_k, x_top_k], axis=-1)
        return x_top_k
        # return Flatten()(x[top_k_indices])

class KMinConcPooling(Layer):
    """
    K-min pooling layer that extracts the k-lowest activations from a sequence (2nd dimension)
    and concatenates the previous bottleneck representation to this score.
    TensorFlow backend.
    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        a, b = input_shape

        return (b[0], (a[2] * self.k), b[-1] + 1)

    def call(self, inputs):
        assert isinstance(inputs, list)
        s, x = inputs
        # swap last two dimensions since top_k will be applied along the last dimension
        # shifted_x = tf.transpose(x, [0, 2, 1])                
        shifted_s = tf.transpose(s, [0, 2, 1])
        neg_shifted_input = tf.scalar_mul(tf.constant(-1, dtype="float32"), shifted_s)

        # extract top_k, returns two tensors [values, indices]
        top_k, top_k_indices = tf.nn.top_k(neg_shifted_input, k=self.k, sorted=True, name=None)
        s_low_k = tf.transpose(top_k, [0, 2, 1])
        s_low_k = tf.scalar_mul(tf.constant(-1, dtype="float32"), s_low_k)
        x_low_k = fetch_top_k_in_x(x, top_k_indices, self.k)
        x_low_k = tf.concat([s_low_k, x_low_k], axis=-1)
        # return flattened output
        return x_low_k


def WeldonConcPooling(s_i, x_i, k):
    max_x_i = KMaxConcPooling(k=k)([s_i, x_i])

    neg_x_i = KMinConcPooling(k=k)([s_i, x_i])

    x_i = Concatenate(axis=-2)([max_x_i, neg_x_i])

    return x_i
