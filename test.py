# import tensorflow as tf
#
# print("TensorFlow version:", tf.__version__)
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print("GPUs detected:", gpus)
# else:
#     print("No GPU detected.")
import importlib.metadata

for dist in importlib.metadata.distributions():
    name = dist.metadata["Name"]
    version = dist.version
    print(f"{name}=={version}")
