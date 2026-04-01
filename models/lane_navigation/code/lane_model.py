import tensorflow as tf

class PiCarNet(tf.keras.Model):
    def __init__(self, image_h = 260, image_w = 260, dropout_first=0.3, dropout_second=0.15):
        super().__init__()
        self.backbone = tf.keras.applications.EfficientNetB2(
            include_top=False, pooling='avg',
            input_shape=(image_h, image_w, 3), weights='imagenet'
        )
        self.drop1    = tf.keras.layers.Dropout(dropout_first)
        self.dense1   = tf.keras.layers.Dense(256, activation='swish')
        self.drop2    = tf.keras.layers.Dropout(dropout_second)
        self.dense2   = tf.keras.layers.Dense(64, activation='swish')
        self.angle_out = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, x, training=False):
        x = self.backbone(x, training=training)
        x = self.drop1(x, training=training)
        x = self.dense1(x)
        x = self.drop2(x, training=training)
        x = self.dense2(x)
        return self.angle_out(x)

    def train_step(self, data):
        imgs, angle_true = data

        with tf.GradientTape() as tape:
            angle_pred, speed_pred = self(imgs, training=True)
            loss = tf.reduce_mean(tf.square(angle_pred - angle_true))

        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)  
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {'loss': loss,}