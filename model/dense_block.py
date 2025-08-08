import tensorflow as tf

class DenseBlock(tf.keras.Model):
    def __init__(self, num_layers, growth_rate=32):
        super(DenseBlock, self).__init__()


        self.num_layers = num_layers
        self.k = growth_rate

        self.bn1_layers = []
        self.conv1_layers = []
        self.dropout1_layers = []
        self.bn2_layers = []
        self.conv2_layers = []
        self.dropout2_layers = []

        for _ in range(num_layers):
            self.bn1_layers.append(tf.keras.layers.BatchNormalization())
            self.conv1_layers.append(tf.keras.layers.Conv2D(filters=4*self.k, kernel_size=(1,1), strides=(1,1), padding='same', activation='relu'))
            self.dropout1_layers.append(tf.keras.layers.Dropout(rate=0.2))
            self.bn2_layers.append(tf.keras.layers.BatchNormalization())
            self.conv2_layers.append(tf.keras.layers.Conv2D(filters=self.k, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
            self.dropout2_layers.append(tf.keras.layers.Dropout(rate=0.2))

    def call(self, inputs, training=False):
        features = [inputs]

        for i in range(self.num_layers):
            x = tf.keras.layers.Concatenate(axis=-1)(features) if len(features) > 1 else features[0]

            # bottleneck
            x = self.bn1_layers[i](x, training)
            x = tf.keras.layers.Activation('relu')(x)
            x = self.conv1_layers[i](x)
            x = self.dropout1_layers[i](x, training)

            x = self.bn2_layers[i](x, training)
            x = tf.keras.layers.Activation('relu')(x)
            x = self.conv2_layers[i](x)
            x = self.dropout2_layers[i](x, training)

            features.append(x)

        return tf.keras.layers.Concatenate(axis=-1)(features)

