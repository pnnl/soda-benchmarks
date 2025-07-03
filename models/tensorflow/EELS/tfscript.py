from tensorflow import keras
import tensorflow as tf
import os
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

params = {
    'INPUT_SHAPE': (240, 1),
    'BATCH_SIZE': 32,
    'LATENT_SIZE': 16,
    'KERNEL_SIZES': [7, 7, 3, 3],
    'FILTER_SIZES': [16, 32, 32, 64],
    'ALPHA': 0.3,
    'DROPOUT': 0.2,
    'LR': 0.001,
}

# Encoder model for EELS data extracted from:
# https://github.com/hollejd1/logicalEELS/blob/b9321874485cfa438c0fedcdb22489975640aa3f/logicalEELS/models.py#L100
def build_and_save_encoder():
    encoderInput = keras.Input(params['INPUT_SHAPE'])
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][0], kernel_size=params['KERNEL_SIZES'][0], strides=2, padding='same')(encoderInput)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][1], kernel_size=params['KERNEL_SIZES'][1], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][2], kernel_size=params['KERNEL_SIZES'][2], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Dropout(params['DROPOUT'])(x)
    x = keras.layers.Conv1D(filters=params['FILTER_SIZES'][3], kernel_size=params['KERNEL_SIZES'][3], strides=2, padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=params['ALPHA'])(x)
    x = keras.layers.Flatten()(x)

    # In the close loop experiment, we may only need z_mean
    z_mean = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_mean')(x)
    z_log_var = keras.layers.Dense(params['LATENT_SIZE'], activation='linear', name='z_log_var')(x)

    encoder = keras.Model(encoderInput, [z_mean, z_log_var], name='encoder')

    # Print model outputs for debugging
    print("Encoder outputs:", [out.name for out in encoder.outputs])
    encoder.summary()
    for i, out in enumerate(encoder.outputs):
        print(f"Output {i}: name={out.name}, shape={out.shape}, dtype={out.dtype}")

    # Save encoder to SavedModel format
    save_path = os.path.join(os.getcwd(), "output/model/eels_encoder/")
    os.makedirs(save_path, exist_ok=True)
    tf.saved_model.save(encoder, save_path)

    # Convert Keras model to ConcreteFunction
    full_model = tf.function(lambda x: encoder(x))
    full_model = full_model.get_concrete_function(
        x=[tf.TensorSpec(encoder.inputs[0].shape, encoder.inputs[0].dtype, name='x1')]
    )

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    tf.io.write_graph(
        graph_or_graph_def=frozen_func.graph,
        logdir=output_dir,
        name="frozen_graph.pbtxt",
        as_text=True
    )

def main():
    save_path = os.path.join(os.getcwd(), "output/model/eels_encoder/")
    if not os.path.exists(save_path) or not os.listdir(save_path):
        build_and_save_encoder()
    else:
        print(f"Model already exists at {save_path}. Skipping export.")

if __name__ == "__main__":
    main()

