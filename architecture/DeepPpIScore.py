import torch
from torch import nn
import torch.nn.functional as F
from architecture.GVPBlock import GVP_embedding
from architecture.GaussianMixture import GM_SV,  GM_SV_V2
from torch_geometric.utils import to_dense_batch


class DeepPpIScore(nn.Module):
    def __init__(self, args):
        super(DeepPpIScore, self).__init__()
        self.args = args

        # encoders
        # peptide encoder
        node_embed_size = args["node_embed_size"]

        self.embed_layer = nn.Sequential(nn.Linear(args["pep_in_feat_size"], node_embed_size), nn.LeakyReLU())
        encoder_layer = nn.TransformerEncoderLayer(d_model=node_embed_size, nhead=args["pep_encoder_n_head"], batch_first=True, 
                                                   dropout=args["dropout_rate"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args["pep_encoder_n_layer"])
    
        # protein encoder
        self.pro_encoder = GVP_embedding(node_in_dim=args["pro_node_in_dim"], node_h_dim=args["pro_node_h_dim"], 
                                         edge_in_dim=args["pro_edge_in_dim"], edge_h_dim=args["pro_edge_h_dim"], 
                                         seq_in=args["pro_seq_in"], num_layers=args["pro_num_layers"], 
                                         drop_rate=args["dropout_rate"]) 
        
        # for cb-cb distance
        self.gm_layer1 = GM_SV(hidden_dim=node_embed_size, n_gaussians=args["n_gaussians"], dropout_rate=args["dropout_rate"])
        # for minimum - minimum distance
        self.gm_layer2 = GM_SV_V2(hidden_dim=node_embed_size, n_gaussians=args["n_gaussians"], dropout_rate=args["dropout_rate"])


    def forward(self, data):
        # protein encoding
        pro_node_s = self.pro_encoder((data['protein']['node_s'], data['protein']['node_v']),
                                                      data[(
                                                          "protein", "p2p", "protein")]["edge_index"],
                                                      (data[("protein", "p2p", "protein")]["edge_s"],
                                                       data[("protein", "p2p", "protein")]["edge_v"]),
                                                      data['protein'].seq)
        # # Ppi15_001, 去掉内部残基编码(col_idx=1), 1958
        # i = 1
        # data['peptide'].node_s = torch.cat((data['peptide'].node_s[:, :i], data['peptide'].node_s[:, i+1:]), dim=1)


        # peptide encoding 
        pep_node_s, pep_node_mask = to_dense_batch(data['peptide'].node_s, data['peptide'].batch, fill_value=0)
        # pep_node_s, pep_node_mask = to_dense_batch(data['peptide'].node_s[:,:677], data['peptide'].batch, fill_value=0)
        # embed
        pep_node_s = self.embed_layer(pep_node_s)
        # transformer encoder, out = [batch_size, seq_len, embed_size]
        pep_node_s = self.transformer_encoder(pep_node_s, src_key_padding_mask=~pep_node_mask) # src_key_padding_mask中padding位置需要用非0值mask

        # 将密集的节点特征张量x_dense(batch_size, max_num_nodes, num_features)和相应的掩码mask(batch_size, max_num_nodes)转换回稀疏的特征张量x和batch格式
        # # method 1. for-loop
        # # 从 pep_node_s 中选取有效的节点
        # pep_node_s_ls = [x[pep_node_mask[i]] for i, x in enumerate(pep_node_s)]
        # pep_node_s = torch.cat(pep_node_s_ls, dim=0)  # 转换为稀疏的 x

        # # 从 mask 生成 batch 张量
        # batch_list = [torch.full((pep_node_mask[i].sum(),), i, dtype=torch.long) for i in range(pep_node_mask.size(0))]
        # batch = torch.cat(batch_list, dim=0)  # 转换为稀疏的 batch

        # method 2. torch.masked_select and torch.repeat_interleave
        # 获取有效（非填充）节点的特征
        pep_node_s = torch.masked_select(pep_node_s, pep_node_mask.unsqueeze(-1)).view(-1, pep_node_s.size(-1))
        # 生成批次信息
        # batch_counts = pep_node_mask.sum(dim=1)
        # _ = torch.repeat_interleave(torch.arange(len(batch_counts), dtype=torch.long), batch_counts)
        
        
        # gm block
        pi1, sigma1, mu1, dist1, batch1 = self.gm_layer1(lig_s=pep_node_s, lig_pos=data['peptide'].X_cb, lig_batch=data['peptide'].batch,
          pro_s=pro_node_s, pro_pos=data['protein'].X_cb, pro_batch=data['protein'].batch, data=data)
        
        pi2, sigma2, mu2, dist2, batch2 = self.gm_layer2(lig_s=pep_node_s, pro_s=pro_node_s,  data=data)
        
        
        return pi1, sigma1, mu1, dist1, batch1, pi2, sigma2, mu2, dist2, batch2
        # pi1: [num_full_contacts, n_gaussians]
        # sigma1: [num_full_contacts, n_gaussians]
        # mu1: [num_full_contacts, n_gaussians]
        # dist1: [num_full_contacts, 1]