import tensorflow as tf
from tensorflow import keras
from keras_visualizer import visualizer 

# Recreate the exact same model, including its weights and the optimizer
new_model = keras.models.load_model('saved_model/comp_model.hdf5')

# Show the model architecture
new_model.summary()
# keras.utils.plot_model(new_model)
visualizer(new_model, format='png', view=True)
