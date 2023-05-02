import torch
import torch.nn as nn



# featuregen version 1
class FeatureGenPytorch(nn.Module):
    def __init__(self):
        super(FeatureGenPytorch, self).__init__()
        self.htriu = torch.tensor([[0] * (bi + 1) + [1] * (20 - bi) for bi in range(21)], dtype = torch.float).unsqueeze(0)
        self.ptriu = torch.tensor([[0] * (bi + 1) + [1] * (24 - bi) for bi in range(25)], dtype = torch.float).unsqueeze(0)
        self.ltriu = torch.tensor([[0] * (bi + 1) + [1] * (19 - bi) for bi in range(20)], dtype = torch.float).unsqueeze(0)
        self.lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            ]
        pass
    
    def forward(self, x):
        x = torch.where(torch.isnan(x), torch.tensor(0.0, dtype=torch.float32), x)

        lefth_x = x[:,40:61,:]
        righth_x = x[:,94:,:]
        pose_x = x[:, 61:86, :]#[:, self.simple_pose]
        lip_x = x[:, :40, :]
        
        lefth_sum = (lefth_x!=0).float().sum()
        righth_sum = (righth_x!=0).float().sum()
        
        cond = lefth_sum > righth_sum
            
        h_x = torch.where(cond, lefth_x, righth_x)
        xfeat = torch.where(cond, torch.cat([lefth_x, pose_x, lip_x], dim = 1), torch.cat([righth_x, pose_x, lip_x], dim = 1) )

        xfeat_xcoordi = xfeat[:, :, 0]
        xfeat_else = xfeat[:, :, 1:]
        xfeat_xcoordi = torch.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
        xfeat = torch.cat([xfeat_xcoordi.unsqueeze(2), xfeat_else], dim = -1)
        
        h_x = h_x.reshape(h_x.shape[0], -1) 
        indices = (h_x.sum(1) != 0)
        if indices.sum() != 0:
            xfeat = xfeat[indices]

        dxyz = torch.cat([xfeat[:-1] - xfeat[1:], torch.zeros(1, xfeat.shape[1], xfeat.shape[2])], dim = 0)
        
        hand = xfeat[:, :21, :3]
        hd = hand.reshape(-1, 21, 1, 3) - hand.reshape(-1, 1, 21, 3)
        hd = torch.sqrt((hd ** 2).sum(-1)) + 1
        hd = hd * self.htriu
        indices = (hd.reshape(hd.shape[0], -1)!=0)
        hd = hd.reshape(hd.shape[0], -1)[indices].reshape(hd.shape[0], -1)
        hdist = hd - 1
        
        pose = xfeat[:, 21:46, :2]
        pd = pose.reshape(-1, 25, 1, 2) - pose.reshape(-1, 1, 25, 2)
        pd = torch.sqrt((pd ** 2).sum(-1)) + 1
        pd = pd * self.ptriu
        indices = (pd.reshape(pd.shape[0], -1)!=0)
        pd = pd.reshape(pd.shape[0], -1)[indices].reshape(pd.shape[0], -1)
        pdist = pd - 1

        olip = xfeat[:, 46:66, :2]
        old = olip.reshape(-1, 20, 1, 2) - olip.reshape(-1, 1, 20, 2)
        old = torch.sqrt((old ** 2).sum(-1)) + 1
        old = old * self.ltriu
        indices = (old.reshape(old.shape[0], -1)!=0)
        old = old.reshape(old.shape[0], -1)[indices].reshape(old.shape[0], -1)
        oldist = old
        oldist = oldist - 1

        ilip = xfeat[:, 66:86, :2]
        ild = ilip.reshape(-1, 20, 1, 2) - ilip.reshape(-1, 1, 20, 2)
        ild = torch.sqrt((ild ** 2).sum(-1)) + 1
        ild = ild * self.ltriu
        indices = (ild.reshape(ild.shape[0], -1)!=0)
        ild = ild.reshape(ild.shape[0], -1)[indices].reshape(ild.shape[0], -1)
        ildist = ild
        ildist = ildist - 1
        
        
        xfeat = torch.cat([
            xfeat[:, :21, :3].reshape(xfeat.shape[0], -1), 
            xfeat[:, 21:46, :2].reshape(xfeat.shape[0], -1), 
            xfeat[:, 46:66, :2].reshape(xfeat.shape[0], -1), 
            dxyz[:, :21, :3].reshape(xfeat.shape[0], -1), 
            dxyz[:, 21:46, :2].reshape(xfeat.shape[0], -1), 
            dxyz[:, 46:66, :2].reshape(xfeat.shape[0], -1), 
            hdist.reshape(xfeat.shape[0], -1),
            pdist.reshape(xfeat.shape[0], -1),
            oldist.reshape(xfeat.shape[0], -1),
            ildist.reshape(xfeat.shape[0], -1),
        ], dim = -1)
        
        xfeat = xfeat[:100]
        #pad_length = 100 - xfeat.shape[0]
        #xfeat = torch.cat([xfeat, torch.zeros(pad_length, xfeat.shape[1])])
        #xfeat = xfeat.reshape(100, 1196)
        
        return xfeat
      

      
# featuregen version 2
class FeatureGenPytorchV2(nn.Module):
    def __init__(self):
        super(FeatureGenPytorchV2, self).__init__()
        self.htriu = torch.tensor([[0] * (bi + 1) + [1] * (20 - bi) for bi in range(21)], dtype = torch.float).unsqueeze(0)
        self.ptriu = torch.tensor([[0] * (bi + 1) + [1] * (24 - bi) for bi in range(25)], dtype = torch.float).unsqueeze(0)
        self.ltriu = torch.tensor([[0] * (bi + 1) + [1] * (19 - bi) for bi in range(20)], dtype = torch.float).unsqueeze(0)
        self.lip_indices = [
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
            291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
            95, 88, 178, 87, 14, 317, 402, 318, 324, 308
            ]
        pass
    
    def forward(self, x):
        x = torch.where(torch.isnan(x), torch.tensor(0.0, dtype=torch.float32), x)

        lefth_x = x[:,40:61,:]
        righth_x = x[:,94:,:]
        pose_x = x[:, 61:86, :]#[:, self.simple_pose]
        lip_x = x[:, :40, :]
        
        lefth_sum = (lefth_x!=0).float().sum()
        righth_sum = (righth_x!=0).float().sum()
        
        cond = lefth_sum > righth_sum
            
        h_x = torch.where(cond, lefth_x, righth_x)
        xfeat = torch.where(cond, torch.cat([lefth_x, pose_x, lip_x], dim = 1), torch.cat([righth_x, pose_x, lip_x], dim = 1) )

        xfeat_xcoordi = xfeat[:, :, 0]
        xfeat_else = xfeat[:, :, 1:]
        xfeat_xcoordi = torch.where(cond, -xfeat_xcoordi, xfeat_xcoordi)
        xfeat = torch.cat([xfeat_xcoordi.unsqueeze(2), xfeat_else], dim = -1)
        
        h_x = h_x.reshape(h_x.shape[0], -1) 
        hand_mask = (h_x.sum(1) != 0)
        token_type_ids = (h_x.sum(1) != 0) + 1

        dxyz = torch.cat([xfeat[:-1] - xfeat[1:], torch.zeros(1, xfeat.shape[1], xfeat.shape[2])], dim = 0)
        
        hand = xfeat[:, :21, :3]
        hd = hand.reshape(-1, 21, 1, 3) - hand.reshape(-1, 1, 21, 3)
        hd = torch.sqrt((hd ** 2).sum(-1)) + 1
        hd = hd * self.htriu
        indices = (hd.reshape(hd.shape[0], -1)!=0)
        hd = hd.reshape(hd.shape[0], -1)[indices].reshape(hd.shape[0], -1)
        hdist = hd - 1
        
        pose = xfeat[:, 21:46, :2]
        pd = pose.reshape(-1, 25, 1, 2) - pose.reshape(-1, 1, 25, 2)
        pd = torch.sqrt((pd ** 2).sum(-1)) + 1
        pd = pd * self.ptriu
        indices = (pd.reshape(pd.shape[0], -1)!=0)
        pd = pd.reshape(pd.shape[0], -1)[indices].reshape(pd.shape[0], -1)
        pdist = pd - 1

        olip = xfeat[:, 46:66, :2]
        old = olip.reshape(-1, 20, 1, 2) - olip.reshape(-1, 1, 20, 2)
        old = torch.sqrt((old ** 2).sum(-1)) + 1
        old = old * self.ltriu
        indices = (old.reshape(old.shape[0], -1)!=0)
        old = old.reshape(old.shape[0], -1)[indices].reshape(old.shape[0], -1)
        oldist = old
        oldist = oldist - 1

        ilip = xfeat[:, 66:86, :2]
        ild = ilip.reshape(-1, 20, 1, 2) - ilip.reshape(-1, 1, 20, 2)
        ild = torch.sqrt((ild ** 2).sum(-1)) + 1
        ild = ild * self.ltriu
        indices = (ild.reshape(ild.shape[0], -1)!=0)
        ild = ild.reshape(ild.shape[0], -1)[indices].reshape(ild.shape[0], -1)
        ildist = ild
        ildist = ildist - 1
        
        
        xfeat = torch.cat([
            xfeat[:, :21, :3].reshape(xfeat.shape[0], -1), 
            xfeat[:, 21:46, :2].reshape(xfeat.shape[0], -1), 
            xfeat[:, 46:66, :2].reshape(xfeat.shape[0], -1), 
            dxyz[:, :21, :3].reshape(xfeat.shape[0], -1), 
            dxyz[:, 21:46, :2].reshape(xfeat.shape[0], -1), 
            dxyz[:, 46:66, :2].reshape(xfeat.shape[0], -1), 
            hdist.reshape(xfeat.shape[0], -1),
            pdist.reshape(xfeat.shape[0], -1),
            oldist.reshape(xfeat.shape[0], -1),
            ildist.reshape(xfeat.shape[0], -1),
            hand_mask.reshape(xfeat.shape[0], -1),
            token_type_ids.reshape(xfeat.shape[0], -1)
        ], dim = -1)
        
        xfeat = xfeat[:200]
        #pad_length = 100 - xfeat.shape[0]
        #xfeat = torch.cat([xfeat, torch.zeros(pad_length, xfeat.shape[1])])
        #xfeat = xfeat.reshape(100, 1196)
        
        return xfeat
