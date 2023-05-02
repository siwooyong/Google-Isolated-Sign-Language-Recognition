import tensorflow as tf
from tensorflow import keras



# featuregen tf version 1
class FeatureGenKeras(keras.Model):
    def __init__(self):
        super(FeatureGenKeras, self).__init__()
        self.htriu = tf.constant([[0] * (bi + 1) + [1] * (20 - bi) for bi in range(21)], dtype = tf.float32)
        self.ptriu = tf.constant([[0] * (bi + 1) + [1] * (24 - bi) for bi in range(25)], dtype = tf.float32)
        self.ltriu = tf.constant([[0] * (bi + 1) + [1] * (19 - bi) for bi in range(20)], dtype = tf.float32)
        self.lip_indices = tf.constant([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            ])

    
    def call(self, x):
        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, dtype=tf.float32), x)
        xfeat = x[:, 468:, :]

        lefth_x = x[:,40:61,:]
        righth_x = x[:,94:,:]
        pose_x = x[:, 61:86, :]#[:, self.simple_pose]
        lip_x = x[:, :40, :]
        
        lefth_sum = tf.reduce_sum(tf.cast(tf.not_equal(lefth_x, 0), dtype=tf.float32))
        righth_sum = tf.reduce_sum(tf.cast(tf.not_equal(righth_x, 0), dtype=tf.float32))
        
        cond = lefth_sum > righth_sum
            
        h_x = tf.where(cond, lefth_x, righth_x)
        xfeat = tf.where(cond, tf.concat([lefth_x, pose_x, lip_x], axis = 1), tf.concat([righth_x, pose_x, lip_x], axis = 1))
        
        xfeat_xcoordi = xfeat[:, :, 0]
        xfeat_else = xfeat[:, :, 1:]
        xfeat_xcoordi = tf.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
        xfeat = tf.concat([xfeat_xcoordi[:, :, tf.newaxis], xfeat_else], axis = -1)
        
        h_x = tf.reshape(h_x, (-1, 21 * 3))
        indices = tf.squeeze(tf.math.reduce_sum(h_x, axis=1) != 0)

        dynamic_size = tf.shape(h_x)[0]
        indices = tf.reshape(indices, (dynamic_size,))

        xfeat = tf.boolean_mask(xfeat, indices)

        dxyz = tf.concat([xfeat[:-1] - xfeat[1:], tf.zeros((1, xfeat.shape[1], xfeat.shape[2]))], axis = 0)
        
        # hand
        hand = xfeat[:, :21, :3]
        hdist = tf.reshape(hand, (-1, 21, 1, 3)) - tf.reshape(hand, (-1, 1, 21, 3))
        hdist = tf.sqrt(tf.reduce_sum(tf.square(hdist), axis=-1)) + 1
        hdist = hdist * self.htriu
        indices = tf.reshape(hdist, (-1, 21 * 21)) != 0
        
        dynamic_size = tf.shape(hdist)[0]
        indices = tf.reshape(indices, (dynamic_size, 21 * 21))
        hdist = tf.boolean_mask(tf.reshape(hdist, (-1, 21 * 21)), indices)
        hdist = hdist - 1
        
        # pose
        pose = xfeat[:, 21:46, :2]
        pdist = tf.reshape(pose, (-1, 25, 1, 2)) - tf.reshape(pose, (-1, 1, 25, 2))
        pdist = tf.sqrt(tf.reduce_sum(tf.square(pdist), axis=-1)) + 1
        pdist = pdist * self.ptriu
        indices = tf.reshape(pdist, (-1, 25 * 25)) != 0
        
        dynamic_size = tf.shape(pdist)[0]
        indices = tf.reshape(indices, (dynamic_size, 25 * 25))
        pdist = tf.boolean_mask(tf.reshape(pdist, (-1, 25 * 25)), indices)
        pdist = pdist - 1
        
        # outlip
        olip = xfeat[:, 46:66, :2]
        oldist = tf.reshape(olip, (-1, 20, 1, 2)) - tf.reshape(olip, (-1, 1, 20, 2))
        oldist = tf.sqrt(tf.reduce_sum(tf.square(oldist), axis=-1)) + 1
        oldist = oldist * self.ltriu
        indices = tf.reshape(oldist, (-1, 20 * 20)) != 0
        
        dynamic_size = tf.shape(oldist)[0]
        indices = tf.reshape(indices, (dynamic_size, 20 * 20))
        oldist = tf.boolean_mask(tf.reshape(oldist, (-1, 20 * 20)), indices)
        oldist = oldist - 1
        
        # inlip
        ilip = xfeat[:, 66:86, :2]
        ildist = tf.reshape(ilip, (-1, 20, 1, 2)) - tf.reshape(ilip, (-1, 1, 20, 2))
        ildist = tf.sqrt(tf.reduce_sum(tf.square(ildist), axis=-1)) + 1
        ildist = ildist * self.ltriu
        indices = tf.reshape(ildist, (-1, 20 * 20)) != 0
        
        dynamic_size = tf.shape(ildist)[0]
        indices = tf.reshape(indices, (dynamic_size, 20 * 20))
        ildist = tf.boolean_mask(tf.reshape(ildist, (-1, 20 * 20)), indices)
        ildist = ildist - 1
        
        xfeat = tf.concat([
            tf.reshape(xfeat[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(xfeat[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(xfeat[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(dxyz[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(dxyz[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(dxyz[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(hdist, [-1, 210]),
            tf.reshape(pdist, [-1, 300]),
            tf.reshape(oldist, [-1, 190]),
            tf.reshape(ildist, [-1, 190]),
        ], axis=-1)
        
        xfeat = xfeat[:100]
        #pad_length = 100 - xfeat.shape[0]
        #xfeat = tf.concat([xfeat, tf.zeros((pad_length, xfeat.shape[1]), dtype=tf.float32)], axis = 0)
        xfeat = tf.reshape(xfeat, (1, -1, 1196))
        
        return xfeat
      
      
      
# featuregen tf version 2
class FeatureGenKerasV2(keras.Model):
    def __init__(self):
        super(FeatureGenKerasV2, self).__init__()
        self.htriu = tf.constant([[0] * (bi + 1) + [1] * (20 - bi) for bi in range(21)], dtype = tf.float32)
        self.ptriu = tf.constant([[0] * (bi + 1) + [1] * (24 - bi) for bi in range(25)], dtype = tf.float32)
        self.ltriu = tf.constant([[0] * (bi + 1) + [1] * (19 - bi) for bi in range(20)], dtype = tf.float32)
        self.lip_indices = tf.constant([
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            ])

    
    def call(self, x):
        x = tf.where(tf.math.is_nan(x), tf.constant(0.0, dtype=tf.float32), x)
        xfeat = x[:, 468:, :]

        lefth_x = x[:,40:61,:]
        righth_x = x[:,94:,:]
        pose_x = x[:, 61:86, :]#[:, self.simple_pose]
        lip_x = x[:, :40, :]
        
        lefth_sum = tf.reduce_sum(tf.cast(tf.not_equal(lefth_x, 0), dtype=tf.float32))
        righth_sum = tf.reduce_sum(tf.cast(tf.not_equal(righth_x, 0), dtype=tf.float32))
        
        cond = lefth_sum > righth_sum
            
        h_x = tf.where(cond, lefth_x, righth_x)
        xfeat = tf.where(cond, tf.concat([lefth_x, pose_x, lip_x], axis = 1), tf.concat([righth_x, pose_x, lip_x], axis = 1))
        
        xfeat_xcoordi = xfeat[:, :, 0]
        xfeat_else = xfeat[:, :, 1:]
        xfeat_xcoordi = tf.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
        xfeat = tf.concat([xfeat_xcoordi[:, :, tf.newaxis], xfeat_else], axis = -1)
        
        h_x = tf.reshape(h_x, (-1, 21 * 3))
        indices = tf.squeeze(tf.math.reduce_sum(h_x, axis=1) != 0)

        dynamic_size = tf.shape(h_x)[0]
        #indices = tf.reshape(indices, (dynamic_size,))

        #xfeat = tf.boolean_mask(xfeat, indices)
        indices = tf.reshape(indices, (dynamic_size,))
        indices = tf.cast(indices, dtype = tf.float32)
        hand_mask = indices + 0.0
        token_type_ids = indices + 1.0

        dxyz = tf.concat([xfeat[:-1] - xfeat[1:], tf.zeros((1, xfeat.shape[1], xfeat.shape[2]))], axis = 0)
        
        # hand
        hand = xfeat[:, :21, :3]
        hdist = tf.reshape(hand, (-1, 21, 1, 3)) - tf.reshape(hand, (-1, 1, 21, 3))
        hdist = tf.sqrt(tf.reduce_sum(tf.square(hdist), axis=-1)) + 1
        hdist = hdist * self.htriu
        indices = tf.reshape(hdist, (-1, 21 * 21)) != 0
        
        dynamic_size = tf.shape(hdist)[0]
        indices = tf.reshape(indices, (dynamic_size, 21 * 21))
        hdist = tf.boolean_mask(tf.reshape(hdist, (-1, 21 * 21)), indices)
        hdist = hdist - 1
        
        # pose
        pose = xfeat[:, 21:46, :2]
        pdist = tf.reshape(pose, (-1, 25, 1, 2)) - tf.reshape(pose, (-1, 1, 25, 2))
        pdist = tf.sqrt(tf.reduce_sum(tf.square(pdist), axis=-1)) + 1
        pdist = pdist * self.ptriu
        indices = tf.reshape(pdist, (-1, 25 * 25)) != 0
        
        dynamic_size = tf.shape(pdist)[0]
        indices = tf.reshape(indices, (dynamic_size, 25 * 25))
        pdist = tf.boolean_mask(tf.reshape(pdist, (-1, 25 * 25)), indices)
        pdist = pdist - 1
        
        # outlip
        olip = xfeat[:, 46:66, :2]
        oldist = tf.reshape(olip, (-1, 20, 1, 2)) - tf.reshape(olip, (-1, 1, 20, 2))
        oldist = tf.sqrt(tf.reduce_sum(tf.square(oldist), axis=-1)) + 1
        oldist = oldist * self.ltriu
        indices = tf.reshape(oldist, (-1, 20 * 20)) != 0
        
        dynamic_size = tf.shape(oldist)[0]
        indices = tf.reshape(indices, (dynamic_size, 20 * 20))
        oldist = tf.boolean_mask(tf.reshape(oldist, (-1, 20 * 20)), indices)
        oldist = oldist - 1
        
        # inlip
        ilip = xfeat[:, 66:86, :2]
        ildist = tf.reshape(ilip, (-1, 20, 1, 2)) - tf.reshape(ilip, (-1, 1, 20, 2))
        ildist = tf.sqrt(tf.reduce_sum(tf.square(ildist), axis=-1)) + 1
        ildist = ildist * self.ltriu
        indices = tf.reshape(ildist, (-1, 20 * 20)) != 0
        
        dynamic_size = tf.shape(ildist)[0]
        indices = tf.reshape(indices, (dynamic_size, 20 * 20))
        ildist = tf.boolean_mask(tf.reshape(ildist, (-1, 20 * 20)), indices)
        ildist = ildist - 1
        
        xfeat = tf.concat([
            tf.reshape(xfeat[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(xfeat[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(xfeat[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(dxyz[:, :21, :3], [-1, 21 * 3]), 
            tf.reshape(dxyz[:, 21:46, :2], [-1, 25 * 2]), 
            tf.reshape(dxyz[:, 46:66, :2], [-1, 20 * 2]), 
            tf.reshape(hdist, [-1, 210]),
            tf.reshape(pdist, [-1, 300]),
            tf.reshape(oldist, [-1, 190]),
            tf.reshape(ildist, [-1, 190]),
            tf.reshape(hand_mask, [-1, 1]),
            tf.reshape(token_type_ids, [-1, 1])
        ], axis=-1)
        
        xfeat = xfeat[:200]
        #pad_length = 200 - xfeat.shape[0]
        #xfeat = tf.concat([xfeat, tf.zeros((pad_length, xfeat.shape[1]), dtype=tf.float32)], axis = 0)
        xfeat = tf.reshape(xfeat, (1, -1, 1198))
        
        return xfeat
