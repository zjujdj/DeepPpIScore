# revised from KarmaDock
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch_cluster
from scipy.spatial import distance_matrix
from MDAnalysis.analysis import distances

three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'X'])}
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']}

esm_aa_mapping = {'LEU': 'L', 'ALA': 'A', 'GLY': 'G', 'VAL': 'V', 'SER': 'S', 'GLU': 'E', 'ARG': 'R', 
 'THR': 'T', 'ILE': 'I', 'ASP': 'D', 'PRO': 'P', 'LYS': 'K', 'GLN': 'Q', 'ASN': 'N', 'PHE': 'F', 
 'TYR': 'Y', 'MET': 'M', 'HIS': 'H', 'TRP': 'W', 'CYS': 'C', 'ASX': 'B', 'SEC': 'U', 'GLX': 'Z', 'PYL': 'O'}
esm_aa_three = list(esm_aa_mapping.keys())

RES_MAX_NATOMS=15  # 文件去H, 重原子数量最多15个


def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF


def obtain_resname(res):
	if res.resname[:2] == "CA":
		resname = "CA"
	elif res.resname[:2] == "FE":
		resname = "FE"
	elif res.resname[:2] == "CU":
		resname = "CU"
	else:
		resname = res.resname.strip()
	
	if resname in METAL:
		return "M"
	else:
		return resname


def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis


def obtain_X_atom_pos(res, name='CA'):
	if obtain_resname(res) == "M":
		return res.atoms.positions[0]
	else:
		try:
			pos = res.atoms.select_atoms(f"name {name}").positions[0]
			return pos
		except:  ##some residues loss the CA atoms
			return res.atoms.positions.mean(axis=0)


def obtain_self_dist(res):
	try:
		#xx = res.atoms.select_atoms("not name H*")
		xx = res.atoms
		dists = distances.self_distance_array(xx.positions)
		ca = xx.select_atoms("name CA")
		c = xx.select_atoms("name C")
		n = xx.select_atoms("name N")
		o = xx.select_atoms("name O")
		return [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
	except:
		return [0, 0, 0, 0, 0]


def calc_res_features(res):
	return np.array(
			obtain_self_dist(res) +  #5
			obtain_dihediral_angles(res) #4		
			)


def check_connect(res_lis, i, j):
    if abs(i-j) == 1 and res_lis[i].segid == res_lis[j].segid:
        return 1
    else:
        return 0


def positional_embeddings_v1(edge_index,
                                num_embeddings=16,
                                period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    # raw
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    # new
    max_relative_feature = 32
    d = torch.clip(d + max_relative_feature, 0, 2 * max_relative_feature)
    d_onehot = F.one_hot(d, 2 * max_relative_feature + 1)
    E = torch.cat((torch.cos(angles), torch.sin(angles), d_onehot), -1)
    return E


def calc_dist(res1, res2):
	dist_array = distances.distance_array(res1.atoms.positions,res2.atoms.positions)
	return dist_array


def obatin_edge(res_lis, src, dst):
    dist = calc_dist(res_lis[src], res_lis[dst])
    return dist.min()*0.1, dst.max()*0.1


def get_orientations(X):
    forward = _normalize(X[1:] - X[:-1])
    backward = _normalize(X[:-1] - X[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def get_sidechains(n, ca, c):
    c, n = _normalize(c - ca), _normalize(n - ca)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec


def get_pocket_feature_mda(pocket_mol, top_k=30):
    with torch.no_grad():
        pure_res_lis, seq, node_s, X_ca, X_cb, X_n, X_c, side_chain_mass, side_chain_mass_mask, X_center_of_mass = [], [], [], [], [], [], [], [], [], []
        for res in pocket_mol.residues:
            try:
                res_name = res.resname.strip()
                res_atoms = res.atoms.select_atoms('not type H')  # 不包含H原子
                dists = distances.self_distance_array(res_atoms.positions)
                # 以下四个原子为主链原子
                ca = res_atoms.select_atoms("name CA")
                c = res_atoms.select_atoms("name C")
                n = res_atoms.select_atoms("name N")
                o = res_atoms.select_atoms("name O")
                intra_dis = [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
                seq.append(three2idx[three2self.get(res_name, 'X')])
                X_ca.append(ca.positions[0])
                X_n.append(n.positions[0])
                X_c.append(c.positions[0])
                node_s.append(intra_dis+obtain_dihediral_angles(res))
                pure_res_lis.append(res)
                try:
                    cb = res_atoms.select_atoms("name CB")
                    X_cb.append(cb.positions[0])
                except:
                    X_cb.append(ca.positions[0])
                # 获取侧链的质心 
                all_atoms_ex_H = res.atoms.select_atoms("not type H")
                main_chain_atoms = res.atoms.select_atoms("name C N CA O")
                side_chain_atoms = all_atoms_ex_H - main_chain_atoms
                if len(side_chain_atoms) > 0:
                    center_of_mass = side_chain_atoms.center_of_mass() 
                    side_chain_mass.append(center_of_mass)
                else: # 侧链质心不存在(GLY), 利用alpha碳原子的位置代替
                    center_of_mass = ca.positions[0]
                    side_chain_mass.append(center_of_mass)
                X_center_of_mass.append(res_atoms.center_of_mass().tolist())
            except:
                continue
        # node features
        seq = torch.from_numpy(np.asarray(seq))  # 残基类型的整数编码
        node_s = torch.from_numpy(np.asarray(node_s))
        # edge features
        X_ca = torch.from_numpy(np.asarray(X_ca))	
        X_cb = torch.from_numpy(np.asarray(X_cb))	
        X_n = torch.from_numpy(np.asarray(X_n))	
        X_c = torch.from_numpy(np.asarray(X_c))	
        # X_center_of_mass = torch.from_numpy(pocket_mol.atoms.center_of_mass(compound='residues'))
        X_center_of_mass = torch.from_numpy(np.asarray(X_center_of_mass))
        edge_index = torch_cluster.knn_graph(X_ca, k=top_k)
        dis_minmax = torch.from_numpy(np.asarray([obatin_edge(pure_res_lis, src, dst) for src, dst in edge_index.T])).view(edge_index.size(1), 2)
        dis_matx_center = distance_matrix(X_center_of_mass, X_center_of_mass)
        cadist = (torch.pairwise_distance(X_ca[edge_index[0]], X_ca[edge_index[1]]) * 0.1).view(-1,1)
        cedist = (torch.from_numpy(dis_matx_center[edge_index[0,:], edge_index[1,:]]) * 0.1).view(-1,1)  # distance between two centers of mass
        edge_connect =  torch.from_numpy(np.asarray([check_connect(pure_res_lis, x, y) for x,y in edge_index.T])).view(-1,1)
        positional_embedding = positional_embeddings_v1(edge_index)  # a sinusoidal encoding of j – i, representing distance along the backbone
        E_vectors = X_ca[edge_index[0]] - X_ca[edge_index[1]]
        edge_s = torch.cat([edge_connect, cadist, cedist, dis_minmax, _rbf(E_vectors.norm(dim=-1), D_count=16, device='cpu'), positional_embedding], dim=1)
        # vector features
        orientations = get_orientations(X_ca)
        sidechains = get_sidechains(n=X_n, ca=X_ca, c=X_c)
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        edge_v = _normalize(E_vectors).unsqueeze(-2)
        xyz_full = torch.from_numpy(np.asarray([np.concatenate([res.atoms.positions[:RES_MAX_NATOMS, :], np.full((max(RES_MAX_NATOMS-len(res.atoms), 0), 3), np.nan)],axis=0) for res in pure_res_lis]))  
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num,(node_s, node_v, edge_s, edge_v))
        # full edge
        full_edge_s = torch.zeros((edge_index.size(1), 5))  # [0, 0, 0, 0, 0]
                                                              # [s, d, t, f, non-cov]
        full_edge_s[edge_s[:, 0]==1, 0] = 1   # edge_s[:, 0]==1 表示共价边
        full_edge_s[edge_s[:, 0]==0, 4] = 1   # edge_s[:, 0]==0 表示非共价边
        full_edge_s = torch.cat([full_edge_s, cadist], dim=-1)

        # 侧链的质心
        side_chain_mass = torch.from_numpy(np.asarray(side_chain_mass))
        # side_chain_mass = torch.nan_to_num(side_chain_mass)

        return (X_ca, X_cb, xyz_full, seq, node_s, node_v, edge_index, edge_s, edge_v, full_edge_s, X_center_of_mass, side_chain_mass)
        # shape:
        # X_ca:[num_res, 3], 每个残基的α碳原子坐标; X_center_of_mass:每个残基的质心坐标, side_chain_mass:每个残基侧链的质心坐标; 
        # xyz_full:[num_res, RES_MAX_NATOMS, 3],每个残基中所有原子的坐标,最大原子数量为24; 
        # seq:[num_res], 残基的类型整数编码; node_s:[num_res, 9], 残基的标量特征; node_v:[num_res, 3, 3], 节点的向量特征; edge_index:[2, num_edges], 边的index; 
        # edge_s:[num_edges, 102], 边的标量特征; edge_v:[num_edges, 1, 3], 边的向量特征; 
        # full_edge_s: [num_edges,6], 边特征, 第1列表示是否为共价边; 第5列表示是否为非共价边; 最后一列表示边的距离


def encode_list(lst):
    # 创建一个空字典来存储每个元素的索引位置
    index_dict = {}
    encoded_list = []

    # 遍历列表并记录每个元素的索引位置
    for i, elem in enumerate(lst):
        if elem not in index_dict:
            index_dict[elem] = [i]  # 记录第一次出现的索引位置
        else:
            index_dict[elem].append(i)  # 记录最后一次出现的索引位置

    # 根据索引位置编码列表
    for i, elem in enumerate(lst):
        if i == index_dict[elem][0]:
            encoded_list.append(0)  # 第一次出现的元素编码为0
        elif i == index_dict[elem][-1]:
            encoded_list.append(1)  # 最后一次出现的元素编码为1
        else:
            encoded_list.append(-1)  # 中间出现的元素编码为-1

    return encoded_list




