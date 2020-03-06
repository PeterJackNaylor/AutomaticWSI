# Defines the neural network used in the WELDON model.
# Implemented in keras

import tensorflow as tf
from keras import backend as K

#config = tf.ConfigProto(log_device_placement=True)
#config.gpu_options.allow_growth = True
#K.tensorflow_backend.set_session(tf.Session(config=config))

from keras.optimizers import Adam
from keras_utils import *
from keras.layers import Add
from nn_metrics import import_metrics

def owkin_model(n_classes, k=10, hidden_fcn=512, weight_decay=0.0005,
                input_depth=2048, drop_out=0.5, gaussian_param=0,
                type_prob="classification"):
    """
    Same model as the one described in the owkin model.
    """

    input_size = (None, input_depth)
    in_layer = Input(shape=input_size)

    if drop_out != 0:
        x_i = Dropout(drop_out, noise_shape=(1, input_depth))(in_layer)
    else:
        x_i = in_layer
    
    if gaussian_param != 0:
        x_i = GaussianDropout(gaussian_param)(x_i)



    x_i = conv_shape_bn_act_drop(x_i, 1, weight_decay, 
                                 drop_out, input_size, name="bottleneck_1",
                                 activation=None)

    x_i = WeldonPooling(x_i, k)
    x_i = dense_bn_act_drop(x_i, hidden_fcn, "fcn_2_1", 
                            weight_decay, drop_out)
    x_i = dense_bn_act_drop(x_i, hidden_fcn, "fcn_2_2", 
                            weight_decay, drop_out)
                            
    output_layer = Dense(n_classes, activation="softmax", use_bias=True,
                         kernel_initializer="glorot_normal",
                         bias_initializer="glorot_uniform",
                         kernel_regularizer=regularizers.l2(weight_decay))(x_i)
    
    model = Model(inputs=in_layer, outputs=output_layer)

    return model

def model_1s_simple(n_classes, hidden_btleneck=128, hidden_fcn=512, 
                    weight_decay=0.0005, input_depth=2048, drop_out=0.5, 
                    aggr="avg", gaussian_param=0, activation_middle="relu"):
    """
    The simplest version of model_1.
    NN architecture:
        Input layer -> bottleneck layer (size hidden_btleneck) -> 1D pooling with aggr
        -> fully connected layer (size hidden_fcn) -> softmax layer (size n_classes)
    Parameters
    ----------
    n_classes : int
        as we have a classification, this the number of neurons (or classes)
        to set the final layer to.
    hidden_btleneck: int
        size of the first intermediate bottleneck layer.
    hidden_fcn: int
        size of the final fully connected dense layers.
    
    options : NameSpace
        NameSpace containing arguments collected by the argumentParser.
    weight_decay: float
        L2 regularisation parameter.
    input_depth: int
        size of the tile that will be fed in to the neural network.
    drop out: float
        drop out value to apply to the input tiles, and to the tissue profile Z_i 
        vector.
    aggr: string, avg or max
        Pooling function to apply to the new tile representation.
    gaussian_param: float
        if 0, does not apply gaussian drop out, if different to 0, we had to the 
        input tile a gaussian noise.
    activation_middle: string, relu, tanh, None,...
        Non linear activation function to apply to the new tile encodings.
    Returns
    -------
    Object: keras.models.Model
        Compiled keras model.
    """
    input_size = (None, input_depth)
    in_layer = Input(shape=input_size)

    if drop_out != 0:
        x_i = Dropout(drop_out, noise_shape=(1, input_depth))(in_layer)
    else:
        x_i = in_layer
    if gaussian_param != 0:
        x_i = GaussianDropout(gaussian_param)(x_i)

    x_i = conv_shape_bn_act_drop(x_i, hidden_btleneck, weight_decay, 
                                 drop_out, input_size, name="bottleneck_1",
                                 activation=activation_middle)
    if aggr == "avg":
        x_i = GlobalAveragePooling1D()(x_i)
    elif aggr == "max":
        x_i = GlobalMaxPooling1D()(x_i)

    x_i = dense_bn_act_drop(x_i, hidden_fcn, "dense", weight_decay, drop_out)

    output_layer = Dense(n_classes, activation="softmax", use_bias=True,
                         kernel_initializer="glorot_normal",
                         bias_initializer="glorot_uniform",
                         kernel_regularizer=regularizers.l2(weight_decay))(x_i)
    
    model = Model(inputs=in_layer, outputs=output_layer)
    return model




def load_model(parameter_dic, options, verbose=True):
    """
    
    Parameters
    ----------
    parameter_dic : dict
        disctionary containing the hyperparameters values.
    options : NameSpace
        NameSpace containing arguments collected by the argumentParser.
    
    Returns
    -------
    Object: keras.models.Model
        Compiled keras model.
    
    Raises
    ------
    ValueError
        Optimizername not known
    """

    if options.y_interest in ["Residual", "Prognostic"]:
        n_classes = 2
    else:
        n_classes = 4

    prob = "classification"
    loss = "categorical_crossentropy"

    input_depth = options.input_depth
    aggr = options.pooling_layer
    k = options.k
    optimizer_name = options.optimizer_name

    hidden_fcn = parameter_dic["hidden_fcn"]
    hidden_btleneck = parameter_dic["hidden_btleneck"]
    gaussian_param = parameter_dic["gaussian_noise"]
    drop_out = parameter_dic["drop_out"]
    weight_decay = parameter_dic["weight_decay"]
    activation_middle = options.activation_middle
    learning_rate = parameter_dic["learning_rate"]

    if options.model == "owkin":
        model = owkin_model(n_classes, k, hidden_fcn, weight_decay,
                            input_depth, drop_out, gaussian_param)
    elif options.model == "model_1S_a":
        model = model_1s_simple(n_classes, hidden_btleneck, hidden_fcn, weight_decay,
                            input_depth, drop_out, aggr, gaussian_param, activation_middle)
    elif options.model == "model_1S_b":
        pass
    elif options.model == "model_1S_c":
        pass
    if verbose:
        print(model.summary())
#        print(config.gpu_options.allow_growth, flush=True)

    if optimizer_name == "Adam":
        opt = Adam(lr=learning_rate, epsilon=1e-08)
    else:
        msg = "Unknown optimizer_name type with name: {}"
        raise ValueError(msg.format(optimizer_name))
    metrics = import_metrics(prob)
    
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=metrics)

    return model
