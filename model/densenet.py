import tensorflow as tf
from dense_block import DenseBlock
from transition_layer import TransitionLayer

class DenseNet(tf.keras.Model):
    def __init__(self, input_shape=(224,223,3), growth_rate=32, include_top=False, weights='imagenet', pooling=None):
        super(DenseNet, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv = tf.keras.layers.Conv2D(filters=2*growth_rate, kernel_size=(7,7), strides=(2,2), padding='same', activation='relu')
        self.relu = tf.keras.layers.Activation('relu')
        self.maxpool = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2), padding='same')
        
        self.dense_block1 = DenseBlock(num_layers=6, growth_rate=growth_rate)
        self.dense_block2 = DenseBlock(num_layers=12, growth_rate=growth_rate)
        self.dense_block3 = DenseBlock(num_layers=24, growth_rate=growth_rate)
        self.dense_block4 = DenseBlock(num_layers=16, growth_rate=growth_rate)

        self.trans1 = TransitionLayer(compression=0.5)
        self.trans2 = TransitionLayer(compression=0.5)
        self.trans3 = TransitionLayer(compression=0.5)

        self.globalavgpool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x, training=False):
        x = self.bn(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense_block1(x, training)
        x = self.trans1(x, training)
        x = self.dense_block2(x, training)
        x = self.trans2(x, training)
        x = self.dense_block3(x, training)
        x = self.trans3(x, training)
        x = self.dense_block4(x, training)
