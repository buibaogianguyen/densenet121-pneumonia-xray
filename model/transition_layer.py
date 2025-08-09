import tensorflow as tf

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, compression):
        super(TransitionLayer, self).__init__()

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.conv = tf.keras.layers.Conv2D(filters=int(compression), kernel_size=(1,1), padding='same', use_bias=False)
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')

    def call(self, x, training=False):
        x = self.bn(x, training)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)

        return x