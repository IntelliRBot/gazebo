from dqn import DQNAgent

import tensorflow_model_optimization as tfmot
import tensorflow as tf

agent = DQNAgent(4,2,load_model=True)

print(agent.model)

converter = tf.lite.TFLiteConverter.from_keras_model(agent.model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_tflite_model = converter.convert()

with open('cartpole_dqn.tflite', 'wb') as f:
    f.write(quantized_tflite_model)