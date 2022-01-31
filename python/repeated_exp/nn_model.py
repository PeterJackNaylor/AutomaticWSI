
# for loading NN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dropout, Activation, BatchNormalization, Dense

from tensorflow.keras import regularizers
from tensorflow.keras import backend as K

def load_model(p, params):
    input_size = (p,)
    input_layer = Input(shape=input_size)
    x_i = input_layer
    hidden_fcn = params[0]
    weight_decay = params[1]
    drop_out = params[2]
    lr = params[3]
    x_i = dense_bn_act_drop(x_i, 
                            hidden_fcn, 
                            "hidden_fcn", 
                            weight_decay, 
                            drop_out)

    output_layer = Dense(2, activation="softmax", use_bias=True,
                         kernel_initializer="glorot_normal",
                         bias_initializer="glorot_uniform",
                         kernel_regularizer=regularizers.l2(weight_decay))(x_i)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    opt = Adam(lr=lr, epsilon=1e-08)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model 

def repeated_experiment(d_train, d_val, d_test, early_stopping):
    x_train, y_train = d_train
    max_epoch = 400
    epoch_patience = 10
    params = [256, 0.005, 0.4, 1e-4]
    model = load_model(x_train.shape[1], params)
    if early_stopping:
        print("ES added")
        es = EarlyStopping(monitor='val_loss', mode='min', 
                        verbose=1, patience=epoch_patience)
        callback = [es]
    else:
        print("No ES")
        callback = None
    h = model.fit(x=x_train, y=y_train, 
                  batch_size=16, 
                  epochs=max_epoch, 
                  verbose=0,
                  validation_data=d_val,
                  shuffle=True,
                  callbacks=callback)
    score = performance(model, d_test)
    n_epochs = len(h.history['val_loss'])
    del model
    K.clear_session()

    return score, n_epochs

def performance(model, data):
    x, y = data
    score = model.evaluate(x, y, verbose=0)[1]
    return score





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