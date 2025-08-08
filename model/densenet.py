import tensorflow as tf
from dense_block import DenseBlock

class DenseNet(tf.keras.Model):
    def __init__(self, input_shape=(224,223,3), growth_rate=32, include_top=False, weights='imagenet', pooling=None):
        super(DenseNet, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv = tf.keras.layers.Conv2D(filters=2*growth_rate, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')

    def call(self, x):
        
