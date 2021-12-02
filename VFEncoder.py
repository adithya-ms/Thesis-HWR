import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from Attention_block import Attention_block

class VFEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
               maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.resnet50 = ResNet50(weights='imagenet', pooling=max, include_top = False)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.attn_layers = [Attention_block(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dense = tf.keras.layers.Dense(d_model)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding resnet features embedding and position encoding.
    x = self.resnet50(x)  # (batch_size, input_seq_len, d_model)
    x = x.squeeze()
    length, width, feature = x.shape[1::]
    x = x.reshape(-1,length*feature,width)

    x = tf.keras.layers.Dense(x)

    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dense(x)
    x = self.layernorm(x)

    for i in range(self.num_layers):
      x = self.attn_layers[i](x, training, mask)
    x = self.dense(x)
    x = self.dropout(x, training=training)

    return x  # (batch_size, input_seq_len, d_model)



