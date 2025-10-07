import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from model.densenet import DenseNet
import numpy as np

model = DenseNet(num_classes=2)
dummy_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
_ = model(dummy_input, training=False)

tf.saved_model.save(model, "onnx_torch_training/saved_model_densenet")