# Google-Isolated-Sign-Language-Recognition


## TLDR<br>
The models used in the competition and the code for data processing are provided by this repository and can be found in the "models" and "data" directories, respectively. The entire training code, including versions for PyTorch and Keras, is available in the "colab" directory. Based on my experiment, it seems that the most significant factors contributing to improved performance were **regularization**, **feature processing** and **embedding_layer**. The final ensemble model consisted of various input features and models, including Transformer, MLP, and GRU.
<br><br>

## Data Processing<br>
Out of the 543 landmarks in the provided raw data, a total of 115 landmarks were used for hands, poses, and lips. Input features were constructed by concatenating xy(z), motion, and distance, resulting in a final input feature size of 1196. Initially, using only the distance feature of hand data was employed, but the significant performance improvement (CV +0.05) was achieved by adding the distance feature of pose and lip data. 

    feature = tf.concat([
            tf.reshape(xyz_hand[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(xyz_pose[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(xyz_lip[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(motion_hand[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(motion_pose[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(motion_lip[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(distance_hand, [-1, 210]),
            tf.reshape(distance_pose, [-1, 300]),
            tf.reshape(distance_outlip, [-1, 190]),
            tf.reshape(distance_inlip, [-1, 190]),
        ], axis=-1)

Additionally, the hand with fewer NaN values was utilized, and the leg landmarks were removed from the pose landmarks. Based on this, two versions of inputs were created. The first version only utilized frames with non-NaN hand data, while the second version included frames with NaN hand data. The former had a max_length of 100, and the latter had a max_length of 200.

    cond = lefth_sum > righth_sum
    h_x = tf.where(cond, lefth_x, righth_x)
    xfeat = tf.where(cond, tf.concat([lefth_x, pose_x, lip_x], axis = 1), tf.concat([righth_x, pose_x, lip_x], axis = 1))

<br><br>

## Augmentation<br>
The hand with fewer NaN values was utilized, and both hands were flipped to be recognized as right hands in the model, which actually contributed to a performance improvement of about 0.01 in CV.

    cond = lefth_sum > righth_sum
    xfeat_xcoordi = xfeat[:, :, 0]
    xfeat_else = xfeat[:, :, 1:]
    xfeat_xcoordi = tf.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
    xfeat = tf.concat([xfeat_xcoordi[:, :, tf.newaxis], xfeat_else], axis = -1)

I did not observe that flip, rotate, mixup, and other augmentation techniques contributed to performance improvement in CV, so I supplemented this by ensembling the models of the two versions of inputs mentioned earlier.
<br><br>

## Model<br>
Prior to being utilized as inputs for the transformer model, the input features, namely xy(z), motion, and distance, underwent individual processing through dedicated embedding layers. Compared to the scenario where features were not processed independently, a performance improvement of 0.01 was observed in the CV score when the features were treated independently.
    
    # embedding layer
    xy = xy_embeddings(xy)
    motion = motion_embeddings(motion)
    distance_hand = distance_hand_embeddings(distance_hand)
    distance_pose = distance_pose_embeddings(distance_pose)
    distance_outlip = distance_outlip_embeddings(distance_outlip)
    distance_inlip = distance_inlip_embeddings(distance_inlip)

    x = tf.concat([xy, motion, distance_hand, distance_pose, distance_outlip, distance_inlip], axis=-1)
    x = relu(x)
    x = fc_layer(x)
    x = TransformerModel(input_ids = None, inputs_embeds=x, attention_mask=x_mask).last_hidden_state
<br>

For Transformer models, I used huggingface's RoBERTa-PreLayerNorm, DeBERTaV2, and GPT2. The input was processed independently for xyz, motion, and distance, and then concatenated to form a 300-dimensional transformer input. The mean, max, and std values of the Transformer output were then concatenated to obtain the final output.

    def get_pool(self, x, x_mask):
        x = x * tf.expand_dims(x_mask, axis=-1)  # apply mask
        nonzero_count = tf.reduce_sum(x_mask, axis=1, keepdims=True)  # count nonzero elements
        max_discount = (1-x_mask)*1e10

        apool = tf.reduce_sum(x, axis=1) / nonzero_count
        mpool = tf.reduce_max(x - tf.expand_dims(max_discount, axis=-1), axis=1)
        spool = tf.sqrt((tf.reduce_sum(((x - tf.expand_dims(apool, axis=1)) ** 2) * tf.expand_dims(x_mask, axis=-1), axis=1) / nonzero_count) + 1e-9)
        return tf.concat([apool, mpool, spool], axis=-1) 

In addition to the Transformer model, simple linear models and GRU models also achieved similar performance to the Transformer model, so I ensembled these three types of models.
<br><br>

## Training<br>

* Scheduler : lr_warmup_cosine_decay 
* Warmup Ratio : 0.2 
* Optimizer : AdamW 
* Weight Decay : 0.01
* Epoch : 40
* Learning Rate : 1e-3 
* Loss Function : CrossEntropyLoss 
* Smoothing Value : 0.65 ~ 0.75

<br><br>

## Regularization<br>
During model training, there are three primary regularization techniques that have made significant contributions to both improving convergence speed and final performance

* Weight normalization applied to the final linear layer([paper](https://arxiv.org/abs/1602.07868))
    
        final_layer = torch.nn.utils.weight_norm(nn.Linear(hidden_size, 250))

* Batch normalization applied before the final linear layer

        x = fc_layer(x)
        x = batchnorm1d(x)
        x = relu(x)
        x = dropout(x)
        x = final_layer(x)

* High weight decay value with the AdamW
<br><br>

## TFLite Conversion<br>
In the early stages of the competition, I worked with PyTorch, which meant I had to deal with numerous errors when converting to TFLite, and ultimately failed to handle dynamic input shapes. In the latter stages of the competition, I began working with Keras, and the number of errors when converting to TFLite was significantly reduced.
<br><br>

## Didn't Work<br>
* Adding the distance feature contributed to performance improvement, but adding angle and direction features did not.
* Increasing the number of transformer layers did not contribute to performance improvement.
* I attempted to model the relationships between landmark points using Transformers or GATs, but the inference speed of the model became slower, and the performance actually decreased.
* Bert-like pretraining (MLM) for XYZ coordinates did not improve performance with the provided data in the competition.
<br><br>

## Learned<br>
I am currently serving in the military in South Korea. During my army training, I learned a lot while coding in between. I was amazed by the culture of Kagglers sharing various ideas and discussing them. It's really cool that something bigger and better something can be born through this community, and I also want to participate in it diligently in the future. Thank you for reading my post.
