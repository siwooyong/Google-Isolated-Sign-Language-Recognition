from transformers import RobertaPreLayerNormConfig, RobertaPreLayerNormModel



# model version 1
class CustomModelV1(nn.Module):
  def __init__(self, args):
    super(CustomModelV1, self).__init__()
    self.args = args

    self.hidden = 384
     
    self.xy_embeddings = nn.Linear(153, self.hidden)
    self.motion_embeddings = nn.Linear(153, self.hidden)
    self.dist_embeddings = nn.Linear(210, self.hidden)
    self.pdist_embeddings = nn.Linear(300, self.hidden)
    self.oldist_embeddings = nn.Linear(190, self.hidden)
    self.ildist_embeddings = nn.Linear(190, self.hidden)
    self.relu = nn.ReLU()
    self.content_embeddings = nn.Linear(self.hidden * 6, self.hidden)
    
    self.encoder = RobertaPreLayerNormModel(
        RobertaPreLayerNormConfig(
            hidden_size = self.hidden,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            intermediate_size = 1024,
            hidden_act = 'relu',
            )
        )
    
    self.fc1 = nn.Linear(self.hidden * 3, 1024)
    self.bn1 = nn.BatchNorm1d(1024)
    self.drop = nn.Dropout(0.4)

    self.out = nn.Linear(1024, 250)
    
    torch.nn.init.xavier_uniform_(self.xy_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.motion_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.dist_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.pdist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.oldist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.ildist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.content_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.fc1.weight)  
    torch.nn.init.xavier_uniform_(self.out.weight)  

  def get_att_mask(self, x):
    att_mask = x.sum(-1)
    att_mask = (att_mask!=0).float()
    return att_mask

  def get_pool(self, x, x_mask):
    x = x * x_mask.unsqueeze(-1)  # apply mask
    nonzero_count = x_mask.sum(1).unsqueeze(-1)  # count nonzero elements
    max_discount = (1-x_mask)*1e10

    apool = x.sum(1) / nonzero_count
    mpool, _ = torch.max(x - max_discount.unsqueeze(-1), dim = 1)
    spool = torch.sqrt((torch.sum(((x - apool.unsqueeze(1)) ** 2)*x_mask.unsqueeze(-1), dim = 1) / nonzero_count)+1e-9)
    return torch.cat([apool, mpool, spool], dim = -1)

  def forward(self, x):
    x_mask = self.get_att_mask(x)

    xy = self.xy_embeddings(x[:, :, :153])
    motion = self.motion_embeddings(x[:, :, 153:306])
    dist = self.dist_embeddings(x[:, :, 306:516])
    pdist = self.pdist_embeddings(x[:, :, 516:816])
    oldist = self.oldist_embeddings(x[:, :, 816:1006])
    ildist = self.ildist_embeddings(x[:, :, 1006:1196])

    x = torch.cat([xy, motion, dist, pdist, oldist, ildist], dim = -1)
    x = self.relu(x)
    x = self.content_embeddings(x)
    x = self.encoder(inputs_embeds = x, attention_mask = x_mask).last_hidden_state
    
    x = self.get_pool(x, x_mask)

    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.drop(x)

    x = self.out(x)
    return x
  
  
  
# model version 2
class CustomModelV2(nn.Module):
  def __init__(self, args):
    super(CustomModelV2, self).__init__()
    self.args = args

    self.hidden = 384
     
    self.xy_embeddings = nn.Linear(153, self.hidden)
    self.motion_embeddings = nn.Linear(153, self.hidden)
    self.dist_embeddings = nn.Linear(210, self.hidden)
    self.pdist_embeddings = nn.Linear(300, self.hidden)
    self.oldist_embeddings = nn.Linear(190, self.hidden)
    self.ildist_embeddings = nn.Linear(190, self.hidden)
    self.relu = nn.ReLU()
    self.content_embeddings = nn.Linear(self.hidden * 6, self.hidden)
    
    self.encoder = RobertaPreLayerNormModel(
        RobertaPreLayerNormConfig(
            hidden_size = self.hidden,
            num_hidden_layers = 1,
            num_attention_heads = 4,
            intermediate_size = 1024,
            hidden_act = 'relu',
            type_vocab_size = 3
            )
        )
    
    self.fc = nn.Linear(self.hidden * 3, 1024)
    self.bn = nn.BatchNorm1d(1024)
    self.drop = nn.Dropout(0.4)

    self.out = nn.Linear(1024, 250)
    
    torch.nn.init.xavier_uniform_(self.xy_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.motion_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.dist_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.pdist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.oldist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.ildist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.content_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.fc.weight)  
    torch.nn.init.xavier_uniform_(self.out.weight)  

  def get_att_mask(self, x):
    att_mask = x.sum(-1)
    att_mask = (att_mask!=0).float()
    return att_mask

  def get_pool(self, x, x_mask):
    x = x * x_mask.unsqueeze(-1)  # apply mask
    nonzero_count = x_mask.sum(1).unsqueeze(-1)  # count nonzero elements
    max_discount = (1-x_mask)*1e10

    apool = x.sum(1) / nonzero_count
    mpool, _ = torch.max(x - max_discount.unsqueeze(-1), dim = 1)
    spool = torch.sqrt((torch.sum(((x - apool.unsqueeze(1)) ** 2)*x_mask.unsqueeze(-1), dim = 1) / nonzero_count)+1e-9)
    return torch.cat([apool, mpool, spool], dim = -1)

  def forward(self, x):
    token_type_ids = x[:, :, -2].long()
    hand_mask = x[:, :, -1].long()
    x = x[:, :, :1196]
    x_mask = self.get_att_mask(x)

    xy = self.xy_embeddings(x[:, :, :153])
    motion = self.motion_embeddings(x[:, :, 153:306])
    dist = self.dist_embeddings(x[:, :, 306:516])
    pdist = self.pdist_embeddings(x[:, :, 516:816])
    oldist = self.oldist_embeddings(x[:, :, 816:1006])
    ildist = self.ildist_embeddings(x[:, :, 1006:1196])

    x = torch.cat([xy, motion, dist, pdist, oldist, ildist], dim = -1)
    x = self.relu(x)
    x = self.content_embeddings(x)
    x = self.encoder(inputs_embeds = x, attention_mask = x_mask, token_type_ids = token_type_ids).last_hidden_state
    
    x = self.get_pool(x, hand_mask)

    x = self.fc(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.drop(x)

    x = self.out(x)
    return x
  
  
  
# model version 3
class CustomModelV3(nn.Module):
  def __init__(self, args):
    super(CustomModelV3, self).__init__()
    self.args = args

    self.hidden = 768
     
    self.xy_embeddings = nn.Linear(153, self.hidden)
    self.motion_embeddings = nn.Linear(153, self.hidden)
    self.dist_embeddings = nn.Linear(210, self.hidden)
    self.pdist_embeddings = nn.Linear(300, self.hidden)
    self.oldist_embeddings = nn.Linear(190, self.hidden)
    self.ildist_embeddings = nn.Linear(190, self.hidden)
    self.relu = nn.ReLU()
    self.content_embeddings = nn.Linear(self.hidden * 6, self.hidden)
    
    self.encoder = nn.Linear(self.hidden, self.hidden)
    
    self.fc1 = nn.Linear(self.hidden * 3, 1024)
    self.bn1 = nn.BatchNorm1d(1024)
    self.drop = nn.Dropout(0.4)

    self.out = nn.Linear(1024, 250)
    
    torch.nn.init.xavier_uniform_(self.xy_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.motion_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.dist_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.pdist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.oldist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.ildist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.content_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.encoder.weight)  
    torch.nn.init.xavier_uniform_(self.fc1.weight)  
    torch.nn.init.xavier_uniform_(self.out.weight)  

  def get_att_mask(self, x):
    att_mask = x.sum(-1)
    att_mask = (att_mask!=0).float()
    return att_mask

  def get_pool(self, x, x_mask):
    x = x * x_mask.unsqueeze(-1)  # apply mask
    nonzero_count = x_mask.sum(1).unsqueeze(-1)  # count nonzero elements
    max_discount = (1-x_mask)*1e10

    apool = x.sum(1) / nonzero_count
    mpool, _ = torch.max(x - max_discount.unsqueeze(-1), dim = 1)
    spool = torch.sqrt((torch.sum(((x - apool.unsqueeze(1)) ** 2)*x_mask.unsqueeze(-1), dim = 1) / nonzero_count)+1e-9)
    return torch.cat([apool, mpool, spool], dim = -1)

  def forward(self, x):
    x_mask = self.get_att_mask(x)

    xy = self.xy_embeddings(x[:, :, :153])
    motion = self.motion_embeddings(x[:, :, 153:306])
    dist = self.dist_embeddings(x[:, :, 306:516])
    pdist = self.pdist_embeddings(x[:, :, 516:816])
    oldist = self.oldist_embeddings(x[:, :, 816:1006])
    ildist = self.ildist_embeddings(x[:, :, 1006:1196])

    x = torch.cat([xy, motion, dist, pdist, oldist, ildist], dim = -1)
    x = self.relu(x)
    x = self.content_embeddings(x)
    x = self.relu(x)
    x = self.encoder(x)
    
    x = self.get_pool(x, x_mask)

    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.drop(x)

    x = self.out(x)
    return x
  
  
  
# model version 4
class CustomModelV4(nn.Module):
  def __init__(self, args):
    super(CustomModelV4, self).__init__()
    self.args = args

    self.hidden = 384
     
    self.xy_embeddings = nn.Linear(153, self.hidden)
    self.motion_embeddings = nn.Linear(153, self.hidden)
    self.dist_embeddings = nn.Linear(210, self.hidden)
    self.pdist_embeddings = nn.Linear(300, self.hidden)
    self.oldist_embeddings = nn.Linear(190, self.hidden)
    self.ildist_embeddings = nn.Linear(190, self.hidden)
    self.relu = nn.ReLU()
    self.content_embeddings = nn.Linear(self.hidden * 6, self.hidden)
    
    self.encoder = nn.GRU(self.hidden, self.hidden, batch_first = True)
    
    self.fc1 = nn.Linear(self.hidden * 3, 1024)
    self.bn1 = nn.BatchNorm1d(1024)
    self.drop = nn.Dropout(0.4)

    self.out = nn.Linear(1024, 250)
    
    torch.nn.init.xavier_uniform_(self.xy_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.motion_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.dist_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.pdist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.oldist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.ildist_embeddings.weight) 
    torch.nn.init.xavier_uniform_(self.content_embeddings.weight)  
    torch.nn.init.xavier_uniform_(self.fc1.weight)  
    torch.nn.init.xavier_uniform_(self.out.weight)  

  def get_att_mask(self, x):
    att_mask = x.sum(-1)
    att_mask = (att_mask!=0).float()
    return att_mask

  def get_pool(self, x, x_mask):
    x = x * x_mask.unsqueeze(-1)  # apply mask
    nonzero_count = x_mask.sum(1).unsqueeze(-1)  # count nonzero elements
    max_discount = (1-x_mask)*1e10

    apool = x.sum(1) / nonzero_count
    mpool, _ = torch.max(x - max_discount.unsqueeze(-1), dim = 1)
    spool = torch.sqrt((torch.sum(((x - apool.unsqueeze(1)) ** 2)*x_mask.unsqueeze(-1), dim = 1) / nonzero_count)+1e-9)
    return torch.cat([apool, mpool, spool], dim = -1)

  def forward(self, x):
    x_mask = self.get_att_mask(x)

    xy = self.xy_embeddings(x[:, :, :153])
    motion = self.motion_embeddings(x[:, :, 153:306])
    dist = self.dist_embeddings(x[:, :, 306:516])
    pdist = self.pdist_embeddings(x[:, :, 516:816])
    oldist = self.oldist_embeddings(x[:, :, 816:1006])
    ildist = self.ildist_embeddings(x[:, :, 1006:1196])

    x = torch.cat([xy, motion, dist, pdist, oldist, ildist], dim = -1)
    x = self.relu(x)
    x = self.content_embeddings(x)
    x, _ = self.encoder(x)
    
    x = self.get_pool(x, x_mask)

    x = self.fc1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.drop(x)

    x = self.out(x)
    return x
