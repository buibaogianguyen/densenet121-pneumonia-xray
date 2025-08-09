import tensorflow as tf

class TransitionLayer(tf.keras.layers.Layer):
    def __init__(self, compression):
        super(TransitionLayer, self).__init__()
        self.compression = compression

        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.Activation('relu')
        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2), padding='same')

    def build(self, input_shape):
        filters = int(input_shape[-1]*self.compression)
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), padding='same', use_bias=False)

    def call(self, x, training=False):
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avgpool(x)

        return x