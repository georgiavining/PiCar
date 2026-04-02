import tensorflow as tf

class PiCarNet(tf.keras.Model):
    def __init__(self, image_h = 120, image_w = 160, dropout_first=0.3, dropout_second=0.15):
        super().__init__()
        self.backbone = tf.keras.applications.MobileNetV3Small(
            include_top=False, pooling='avg',
            input_shape=(image_h, image_w, 3), weights='imagenet'
        )
        
        self.head = tf.keras.Sequential([
            tf.keras.layers.Dropout(dropout_first),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(dropout_second),
            tf.keras.layers.Dense(1, activation='sigmoid')  
        ])

    def call(self, x, training=False):
        x = self.backbone(x, training=training)
        x = self.head(x, training=training)
        return x