from transformers import TFRobertaPreLayerNormModel, TFDebertaV2Model, TFGPT2Model, RobertaPreLayerNormConfig, DebertaV2Config, GPT2Config

class CustomModel(keras.Model):
    def __init__(self, args):
        super(CustomModel, self).__init__()
        
        self.args = args
        self.hidden = 300

        self.xy_embeddings = keras.layers.Dense(units=self.hidden, name="xy_embeddings")
        self.motion_embeddings = keras.layers.Dense(units=self.hidden, name="motion_embeddings")
        self.hdist_embeddings = keras.layers.Dense(units=self.hidden, name="hdist_embeddings")
        self.pdist_embeddings = keras.layers.Dense(units=self.hidden, name="pdist_embeddings")
        self.oldist_embeddings = keras.layers.Dense(units=self.hidden, name="oldist_embeddings")
        self.ildist_embeddings = keras.layers.Dense(units=self.hidden, name="ildist_embeddings")
        self.relu = keras.layers.ReLU()
        self.content_embeddings = keras.layers.Dense(units=self.hidden, name="content_embeddings")
        
        if args == 'tfrobertaprelayernorm':
          self.encoder = TFRobertaPreLayerNormModel(
              RobertaPreLayerNormConfig(
                  hidden_size = self.hidden,
                  num_hidden_layers = 1,
                  num_attention_heads = 4,
                  intermediate_size = 900,
                  hidden_act = 'relu',
                  vocab_size = 3, 
                  ),
                  name="encoder"
                  )
        elif args == 'tfdebertav2':
          self.encoder = TFDebertaV2Model(
              DebertaV2Config(
                  hidden_size = self.hidden,
                  num_hidden_layers = 1,
                  num_attention_heads = 4,
                  intermediate_size = 900,
                  hidden_act = 'relu',
                  vocab_size = 3, 
                  ),
                  name="encoder"
                  )
        
        self.fc = keras.layers.Dense(units=1024, name="fc")
        self.bn = keras.layers.BatchNormalization(name="bn")
        self.relu = keras.layers.ReLU()
        self.drop = keras.layers.Dropout(rate=0.4, name="drop")

        self.out = keras.layers.Dense(units=250, activation='softmax', name="out")

        self.xy_embeddings.kernel_initializer = 'glorot_uniform'
        self.motion_embeddings.kernel_initializer = 'glorot_uniform'
        self.hdist_embeddings.kernel_initializer = 'glorot_uniform'
        self.pdist_embeddings.kernel_initializer = 'glorot_uniform'
        self.oldist_embeddings.kernel_initializer = 'glorot_uniform'
        self.ildist_embeddings.kernel_initializer = 'glorot_uniform'
        self.content_embeddings.kernel_initializer = 'glorot_uniform'
        self.fc.kernel_initializer = 'glorot_uniform'
        self.out.kernel_initializer = 'glorot_uniform'

    def get_att_mask(self, x):
        att_mask = tf.math.reduce_sum(x, axis=-1)
        att_mask = tf.cast(tf.math.not_equal(att_mask, 0), tf.float32)
        return att_mask

    def get_pool(self, x, x_mask):
        x = x * tf.expand_dims(x_mask, axis=-1)  # apply mask
        nonzero_count = tf.reduce_sum(x_mask, axis=1, keepdims=True)  # count nonzero elements
        max_discount = (1-x_mask)*1e10

        apool = tf.reduce_sum(x, axis=1) / nonzero_count
        mpool = tf.reduce_max(x - tf.expand_dims(max_discount, axis=-1), axis=1)
        spool = tf.sqrt((tf.reduce_sum(((x - tf.expand_dims(apool, axis=1)) ** 2) * tf.expand_dims(x_mask, axis=-1), axis=1) / nonzero_count) + 1e-9)
        return tf.concat([apool, mpool, spool], axis=-1)

    def call(self, x):
        x_mask = self.get_att_mask(x)

        xy = self.xy_embeddings(x[:, :, :153])
        motion = self.motion_embeddings(x[:, :, 153:306])
        dist = self.hdist_embeddings(x[:, :, 306:516])
        pdist = self.pdist_embeddings(x[:, :, 516:816])
        oldist = self.oldist_embeddings(x[:, :, 816:1006])
        ildist = self.ildist_embeddings(x[:, :, 1006:1196])

        x = tf.concat([xy, motion, dist, pdist, oldist, ildist], axis=-1)
        x = self.relu(x)
        x = self.content_embeddings(x)
        x = self.encoder(input_ids = None, inputs_embeds=x, attention_mask=x_mask).last_hidden_state

        x = self.get_pool(x, x_mask)

        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)

        x = self.out(x)
        return x
