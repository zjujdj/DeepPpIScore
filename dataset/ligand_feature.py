import pickle
import pandas as pd
import MDAnalysis as mda
import esm
from datetime import datetime
import torch
import numpy as np

RES_MAX_NATOMS=15  # 不考虑H,最多15个重原子
three2idx = {k:v for v, k in enumerate(['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'X'])}
Standard_AAMAP = {'HIS': 'H', 'ASP': 'D', 'ARG': 'R', 'PHE': 'F', 'ALA': 'A', 'CYS': 'C', 'GLY': 'G', 'GLN': 'Q',
                  'GLU': 'E', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'SER': 'S', 'TYR': 'Y', 'THR': 'T',
                  'ILE': 'I', 'TRP': 'W', 'PRO': 'P', 'VAL': 'V'}
three2self = {v:v for v in ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR', 
										'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP', 
										'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
										'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP']}
esm_aa_mapping = {'LEU': 'L', 'ALA': 'A', 'GLY': 'G', 'VAL': 'V', 'SER': 'S', 'GLU': 'E', 'ARG': 'R', 
 'THR': 'T', 'ILE': 'I', 'ASP': 'D', 'PRO': 'P', 'LYS': 'K', 'GLN': 'Q', 'ASN': 'N', 'PHE': 'F', 
 'TYR': 'Y', 'MET': 'M', 'HIS': 'H', 'TRP': 'W', 'CYS': 'C', 'ASX': 'B', 'SEC': 'U', 'GLX': 'Z', 'PYL': 'O'}
esm_aa_three = list(esm_aa_mapping.keys())

standd_aa_thr = list(Standard_AAMAP.keys())
standd_aa_one = list(Standard_AAMAP.values())

AABLOSUM62 = pd.read_csv('../data/AAD/data/AABLOSUM62.csv')
AAindex = pd.read_csv('../data/AAD/data/AAindex.csv')
AAMOE2D = pd.read_csv('../data/AAD/data/AAMOE2D.csv')

# 刪除含有na值的列
AABLOSUM62.dropna(axis=1, inplace=True)  # len=20
AAMOE2D.dropna(axis=1, inplace=True)  # len=148
# 刪除含有na值的行
AAindex.dropna(axis=0,inplace=True)  # len=508
AAindex = AAindex.iloc[:, 7:]  

AABLOSUM62_dic, AAindex_dic, AAMOE2D_dic  = {}, {}, {}
for idx, key in enumerate(AABLOSUM62.iloc[:, 0].tolist()):
    AABLOSUM62_dic[key] = AABLOSUM62.iloc[idx, 1:].tolist()
for idx, key in enumerate(AAMOE2D.iloc[:, 0].tolist()):
    AAMOE2D_dic[key] = AAMOE2D.iloc[idx, 1:].tolist()
for col in AAindex.columns:
    AAindex_dic[col] = AAindex[col].tolist()

# esm_rep_pkl_file = '../data/pep_esm_rep.pkl'
# with open(esm_rep_pkl_file, 'rb') as f:
#     esm_rep = pickle.load(f)
# esm_rep_keys = list(esm_rep.keys())

esm_rep, esm_rep_keys = [], []

def get_seq(u):
    # u = mda.Universe(pep_pdb_file)
    seq = []
    for res in u.residues:
        res_name = res.resname.strip()
        if res_name in esm_aa_three:
            seq.append(esm_aa_mapping[res_name])
        else:
            seq.append('X')
    return ''.join(seq)

def obtain_dihediral_angles(res):
    angle_lis = [0, 0, 0, 0]
    for idx, angle in enumerate([res.phi_selection, res.psi_selection, res.omega_selection, res.chi1_selection]):
        try:
            angle_lis[idx] = angle().dihedral.value()
        except:
            continue
    return angle_lis


from MDAnalysis.analysis import distances
from .protein_feature import get_orientations, get_sidechains
def get_pep_corrd_feat(pep_mol, pdb_id):
    # 标准多肽序列表征:  标准残基类型的整型编码(1) + BLOSUM62(20) + AAindex(最多554, 508) + AAMOE2D(148) + ESMFOLD(1280)
    # pdb_id_chain = '1abo_C'
    pure_res_lis, X_ca, X_cb, side_chain_mass, torsion, X_center_of_mass = [], [], [], [], [], []
    pro_rep_node_s, pro_rep_node_v, X_n, X_c = [], [], [], []

    for res in pep_mol.residues:
        res_name = res.resname.strip()
        res_atoms = res.atoms.select_atoms('not type H')  # 不包含H原子
        # 主链原子alpha碳原子
        ca = res_atoms.select_atoms("name CA")
        X_ca.append(ca.positions[0])
        pure_res_lis.append(res)
        
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
        
        # 
        try:
            cb = res_atoms.select_atoms("name CB")
            X_cb.append(cb.positions[0])
        except:
            X_cb.append(ca.positions[0])
        
        torsion.append(obtain_dihediral_angles(res))

        X_center_of_mass.append(res_atoms.center_of_mass().tolist())

        # 获取三维特征，node_s, node_v, 来源于蛋白口袋残基的表征
        # ca = res_atoms.select_atoms("name CA")
        c = res_atoms.select_atoms("name C")
        n = res_atoms.select_atoms("name N")
        o = res_atoms.select_atoms("name O")
        # X_ca.append(ca.positions[0])
        X_n.append(n.positions[0])
        X_c.append(c.positions[0])
        
        dists = distances.self_distance_array(res_atoms.positions)
        intra_dis = [dists.max()*0.1, dists.min()*0.1, distances.dist(ca,o)[-1][0]*0.1, distances.dist(o,n)[-1][0]*0.1, distances.dist(n,c)[-1][0]*0.1]
        pro_rep_node_s.append(intra_dis+obtain_dihediral_angles(res))
    
    pro_rep_node_s = torch.from_numpy(np.asarray(pro_rep_node_s))
    X_ca = torch.from_numpy(np.asarray(X_ca))	
    X_n = torch.from_numpy(np.asarray(X_n))	
    X_c = torch.from_numpy(np.asarray(X_c))	
    orientations = get_orientations(X_ca)
    sidechains = get_sidechains(n=X_n, ca=X_ca, c=X_c)
    pro_rep_node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    pro_rep_node_s, pro_rep_node_v = map(torch.nan_to_num,(pro_rep_node_s, pro_rep_node_v))
        

    # 侧链的质心
    side_chain_mass = torch.from_numpy(np.asarray(side_chain_mass))
    side_chain_mass = torch.nan_to_num(side_chain_mass)

    # alpha C的位置
    # X_ca = torch.from_numpy(np.asarray(X_ca))

    X_cb = torch.from_numpy(np.asarray(X_cb))

    torsion = torch.from_numpy(np.asarray(torsion))

    # 残基质心
    # X_center_of_mass = torch.from_numpy(pep_mol.atoms.center_of_mass(compound='residues'))
    X_center_of_mass = torch.from_numpy(np.asarray(X_center_of_mass))

    xyz_full = torch.from_numpy(np.asarray([np.concatenate([res.atoms.positions[:RES_MAX_NATOMS, :], np.full((max(RES_MAX_NATOMS-len(res.atoms), 0), 3), np.nan)],axis=0) for res in pure_res_lis]))

    return X_ca, X_cb, X_center_of_mass, side_chain_mass, pure_res_lis, xyz_full, torsion, pro_rep_node_s, pro_rep_node_v
    # [num_res, 3],  [num_res, 3],  [num_res, 3], [num_res, 1957], [num_res]


esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local("../data/esm2_t33_650M_UR50D.pt")
batch_converter = alphabet.get_batch_converter()
# esm_model, alphabet = None, None
# batch_converter = None
def get_pep_node_feat(pep_mol, pdb_id):
    # 标准多肽序列表征:  标准残基类型的整型编码(1) + BLOSUM62(20) + AAindex(最多554, 508) + AAMOE2D(148) + ESMFOLD(1280)
    # pdb_id_chain = '1abo_C'
    seq, aa_feat, res_intra_idxs = [], [], []

    for idx, res in enumerate(pep_mol.residues):
        if idx == 0:
            res_intra_idxs.append(1)
        elif idx == len(pep_mol.residues) - 1:
            res_intra_idxs.append(2)
        else:
            res_intra_idxs.append(-1)

        res_name = res.resname.strip()
        # seq.append(standd_aa_thr.index(res_name)+1)
        seq.append(three2idx[three2self.get(res_name, 'X')])

        res_name_one_letter = Standard_AAMAP[res_name]
        aa_feat.append(AABLOSUM62_dic[res_name_one_letter] + AAindex_dic[res_name_one_letter] + AAMOE2D_dic[res_name_one_letter])
        
    # seq features
    seq = torch.from_numpy(np.asarray(seq))  # 残基类型的整数编码
    seq = seq.view(-1, 1)  # shape = [num_res, 1]

    # 
    res_intra_idxs = torch.from_numpy(np.asarray(res_intra_idxs))
    res_intra_idxs = res_intra_idxs.view(-1, 1)

    # aa_feat
    aa_feat = torch.from_numpy(np.asarray(aa_feat))  # shape = [num_res, 676]

    # esm_feat
    if pdb_id in esm_rep_keys:
        esm_feat = torch.from_numpy(esm_rep[pdb_id][1])  # shape = [num_res, 1280]
    # 需要重新计算esmfold2的表征
    else:
        seq_esm = get_seq(pep_mol)
        seq_data = [(pdb_id, seq_esm)]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        model = esm_model.eval()
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representations = results["representations"][33]
        # 1表示padding, 1:-1 -> 起始字符:结束字符, 29表示'.', gap
        esm_feat = token_representations[0][(batch_tokens[0] != 1) & (batch_tokens[0] != 29)][1:-1]

        # # 获取当前时间
        # # 更新序列的esm数据
        # esm_rep[pdb_id] = (seq_esm, esm_feat)
        # rep_pkl = '../PpI/data/pep_esm_rep_%s.pkl' % (str(datetime.now()).replace(' ', '_'))
        # with open(rep_pkl, 'wb') as f:
        #     pickle.dump(esm_rep, f)

    node_s = torch.cat([seq, res_intra_idxs, aa_feat, esm_feat], dim=1)

    return node_s
    # [num_res, 3],  [num_res, 3],  [num_res, 3], [num_res, 1957], [num_res]
