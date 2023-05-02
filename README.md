# Google-Isolated-Sign-Language-Recognition

**TLDR**<br>
In my experiment, it appears that feature processing had the greatest impact on performance. The final ensemble model consisted of various input features and models, including Transformer, MLP, and GRUs.
<br><br>

**Data processing**<br>
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

**Augmentation**<br>
The hand with fewer NaN values was utilized, and both hands were flipped to be recognized as right hands in the model, which actually contributed to a performance improvement of about 0.01 in CV.

    cond = lefth_sum > righth_sum
    xfeat_xcoordi = xfeat[:, :, 0]
    xfeat_else = xfeat[:, :, 1:]
    xfeat_xcoordi = tf.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
    xfeat = tf.concat([xfeat_xcoordi[:, :, tf.newaxis], xfeat_else], axis = -1)

I did not observe that flip, rotate, mixup, and other augmentation techniques contributed to performance improvement in CV, so I supplemented this by ensembling the models of the two versions of inputs mentioned earlier.
<br><br>

**Model**<br>
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

**Training**<br>
I utilized an lr_warmup_cosine_decay scheduler with warmup ratio 0.2 and an AdamW optimizer with a weight decay value of 0.01. The total number of epochs was 40, and the learning rate was set to 1e-3. A label smoothing value of 0.65 exhibited the best performance, which might be attributed to the noise in the input that was used.
<br><br>

**TFLite Conversion**<br>
In the early stages of the competition, I worked with PyTorch, which meant I had to deal with numerous errors when converting to TFLite, and ultimately failed to handle dynamic input shapes. In the latter stages of the competition, I began working with Keras, and the number of errors when converting to TFLite was significantly reduced.
<br><br>

**Didn't work**<br>
1. Adding the distance feature contributed to performance improvement, but adding angle and direction features did not.
2. Increasing the number of transformer layers did not contribute to performance improvement.
3. I attempted to model the relationships between landmark points using Transformers or GATs, but the inference speed of the model became slower, and the performance actually decreased.
4. Pretraining with the data provided in the competition did not improve performance.
<br><br>

**Learned**<br>
I am currently serving in the military in South Korea. During my army training, I learned a lot while coding in between. I was amazed by the culture of Kagglers sharing various ideas and discussing them. It's really cool that something bigger and better something can be born through this community, and I also want to participate in it diligently in the future. Thank you for reading my post.
