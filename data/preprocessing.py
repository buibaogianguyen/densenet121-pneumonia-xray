import tensorflow as tf

class Preprocessor(tf.keras.layers.Layer):
    def __init__(self, img_shape=(224,224), augment=False):
        super().__init__()

        self.img_shape = img_shape

        if augment:
            self.augment = tf.keras.Sequential([
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.1),
                tf.keras.layers.RandomZoom(0.2),
                tf.keras.layers.RandomContrast(0.2),
                tf.keras.layers.RandomTranslation(0.1, 0.1)
            ])
        else:
            self.augment = None

        self.normalize = lambda x: tf.cast(x, tf.float32) / 255.0
        
    def call(self, img, training=False):
        img = tf.image.resize(img, self.img_shape)

        if training == True and self.augment is not None:
            img = self.augment(img)

        img = self.normalize(img)

        return img
