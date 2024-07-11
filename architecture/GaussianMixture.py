'''
@Author  :   revised from DeepDock
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch_geometric.utils import to_dense_batch

RES_MAX_NATOMS=15  # 文件去H, 重原子数量最多15个

class GM_SV(nn.Module):
    def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, dist_threhold=1000):
        super(GM_SV, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)  
        self.dist_threhold = dist_threhold
    
    def forward(self, lig_s, lig_pos, lig_batch, pro_s, pro_pos, pro_batch, data):
        
        h_l_x, l_mask = to_dense_batch(lig_s, lig_batch, fill_value=0)  # h_l_x: [batch_size, max_num_nodes, feat_size]
        h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)  # t_mask: [batch_size, max_num_nodes]
        h_l_pos, _ = to_dense_batch(lig_pos, lig_batch, fill_value=0)  # h_l_pos: [batch_size, max_num_nodes, 3]
        h_t_pos, _ = to_dense_batch(pro_pos, pro_batch, fill_value=0)
        
        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t
        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]


        C = h_l_x + h_t_x  # added by jdj 20230911; 两个节点的表征相加
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(lig_s.device)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)  # 权重
        sigma = F.elu(self.z_sigma(C))+1.1  # 方差, +1.1 保证方差大于0
        mu = F.elu(self.z_mu(C))+1  # 均值
        
        dist = self.compute_euclidean_distances_matrix(h_l_pos, h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]  # 与下一句等价
        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch
    
    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        return dists**0.5


class GM_SV_V2(nn.Module):
    '''
    计算res-res之间的最小距离
    '''
    def __init__(self, hidden_dim, n_gaussians, dropout_rate=0.15, dist_threhold=1000):
        super(GM_SV_V2, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ELU(), nn.Dropout(p=dropout_rate)) 
        self.z_pi = nn.Linear(hidden_dim, n_gaussians)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians)  
        self.dist_threhold = dist_threhold
    
    def forward(self, lig_s, pro_s, data):
        lig_batch, pro_batch = data['peptide'].batch, data['protein'].batch
        h_l_x, l_mask = to_dense_batch(lig_s, lig_batch, fill_value=0)  # h_l_x: [batch_size, self.N_l, feat_size]
        h_t_x, t_mask = to_dense_batch(pro_s, pro_batch, fill_value=0)  # t_mask: [batch_size, self.N_t]
        h_l_pos, _ = to_dense_batch(data['peptide'].xyz_full, lig_batch, fill_value=0)  # h_l_pos: [batch_size, self.N_l, 24, 3] -> 每个残基最多24个原子
        h_t_pos, _ = to_dense_batch(data['protein'].xyz_full, pro_batch, fill_value=0)
        
        assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
        (B, N_l, C_out), N_t = h_l_x.size(), h_t_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_t = N_t
        # Combine and mask
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_t, 1) # [B, N_l, N_t, C_out]

        h_t_x = h_t_x.unsqueeze(-3)
        h_t_x = h_t_x.repeat(1, N_l, 1, 1) # [B, N_l, N_t, C_out]

        C = h_l_x + h_t_x  # added by dejunjiang 20230911; 两个节点的表征相加
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        # Get batch indexes for ligand-target combined features
        C_batch = torch.tensor(range(B)).unsqueeze(-1).unsqueeze(-1).to(lig_s.device)
        C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
        
        # Outputs
        pi = F.softmax(self.z_pi(C), -1)  # 权重
        sigma = F.elu(self.z_sigma(C))+1.1  # 方差, +1.1 保证方差大于0
        mu = F.elu(self.z_mu(C))+1  # 均值  
        dist = self.compute_euclidean_distances_matrix(h_l_pos.view(h_l_pos.size(0), -1, 3), h_t_pos.view(h_t_pos.size(0), -1, 3))[C_mask]  # -> [batch_size, self.N_l*24, 3], [batch_size, self.N_t*24, 3]
        return pi, sigma, mu, dist.unsqueeze(1).detach(), C_batch


    def compute_euclidean_distances_matrix(self, X, Y):
        # Based on: https://medium.com/@souravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c
        # (X-Y)^2 = X^2 + Y^2 -2XY
        X = X.double()
        Y = Y.double()

        dists = -2 * torch.bmm(X, Y.permute(0, 2, 1)) + torch.sum(Y**2, axis=-1).unsqueeze(1) + torch.sum(X**2, axis=-1).unsqueeze(-1)
        dists = dists.view(self.B, self.N_l, RES_MAX_NATOMS, self.N_t, RES_MAX_NATOMS)
        dists_min1, _ = torch.min(torch.nan_to_num(dists**0.5, 10000), dim=-1)
        dists_min2, _ = torch.min(dists_min1, dim=2)
        return dists_min2
        # [batch_size, self.N_l, self.N_t]

