import tensorflow as tf

# Set GPU 0 as the visible device
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use GPU 0
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("Using GPU 0:", gpus[0])
    except RuntimeError as e:
        print(e)
else:
    print("No GPUs found. Make sure the correct drivers are installed.")
