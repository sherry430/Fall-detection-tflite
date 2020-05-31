import tensorflow as tf
from tensorflow.python.platform import gfile
model_path = "./frozen.pb"

# read graph definition
f = gfile.FastGFile(model_path, "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())

convert = tf.lite.TFLiteConverter.from_frozen_graph("./frozen.pb", input_arrays=["input_dataset/input_dataset_x"], output_arrays=["fc2/add"],
                                                  input_shapes={"input_dataset/input_dataset_x": [None, 1200]})
convert.post_training_quantize  = True
tflite_model=convert.convert()
open("model.tflite", "wb").write(tflite_model)
print("finish!")
