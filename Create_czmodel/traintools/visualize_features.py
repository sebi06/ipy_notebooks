import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.keras.models import load_model, Model
import numpy as np
import matplotlib.pyplot as plt

from data import load_prediction_paths, load_prediction_data

if __name__ == '__main__':

    # load input image and model path
    input_path, model_path = load_prediction_paths()
    data, filelist = load_prediction_data(input_path)

    print(data.shape)

    model = load_model(model_path)
    conv_layer_names = [layer.name for layer in model.layers if
                       (("conv2d" in layer.name) and not ("transpose" in layer.name))][:-1]
    conv_layer_outputs = [layer.output for layer in model.layers if
                          (("conv2d" in layer.name) and not ("transpose" in layer.name))][:-1]

    model = Model(inputs = model.inputs, outputs = conv_layer_outputs)
    predicted_layers = model.predict(np.expand_dims(data[0], axis=0))

    for layer, name in zip(predicted_layers, conv_layer_names):
        fig = plt.figure(figsize=(12,12))
        fig.suptitle(name + " " + str(layer.shape))
        for i in range(16):
            ax = plt.subplot(4,4,i+1)
            ax.imshow(layer[0,:,:,i], cmap="gray")
            ax.axis("off")
        plt.show()