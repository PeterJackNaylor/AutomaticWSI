from keras.applications.resnet import ResNet50

def prep_model(weight=None):
    """
    Loads a model for tile encoding. This function
    can optionnaly take a different weight path to load.
    Parameters
    ----------
    weight: string, 
        path to weight folder
    Returns
    -------
    A keras model to encode.
    """
    shape = (224, 224, 3)
    model = ResNet50(include_top=False, 
                     weights="imagenet", 
                     input_shape=shape, 
                     pooling='avg')
    if weight != "imagenet":
        print('loading')
        model.load_weights(weight, by_name=True)
    return model
