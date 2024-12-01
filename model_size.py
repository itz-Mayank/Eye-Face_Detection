import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras import regularizers

# Load your pre-existing model
model = load_model('face_eye_liveness_model_lccfasd.h5')

# 1. Disable mixed precision globally (force float32)
tf.keras.mixed_precision.set_global_policy('float32')

# 2. Apply Pruning
def apply_pruning(model):
    # Define the pruning schedule (PolynomialDecay or other types)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,  # Start pruning at 0%
        final_sparsity=0.5,    # End pruning at 50%
        begin_step=0,          # When to start pruning
        end_step=1000          # When to stop pruning (adjust as needed)
    )

    # Apply pruning to each Dense layer
    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=lambda layer: tfmot.sparsity.keras.prune_low_magnitude(layer, pruning_schedule)
    )

    # Ensure that all layers are using float32 (to avoid mixed precision issues)
    for layer in pruned_model.layers:
        if isinstance(layer, Dense):
            # Make sure to cast the weights and bias to float32
            layer.kernel = tf.cast(layer.kernel, dtype=tf.float32)
            if layer.bias is not None:
                layer.bias = tf.cast(layer.bias, dtype=tf.float32)

    # Recompile the pruned model
    pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return pruned_model

# Apply pruning to your model
pruned_model = apply_pruning(model)

# 3. Save the pruned model
pruned_model.save('pruned_model.h5')
print("Pruned model saved as 'pruned_model.h5'")

# 4. Optionally, you can evaluate the model after pruning (if you have a validation set)
# pruned_model.evaluate(x_val, y_val)  # Provide your validation data here
