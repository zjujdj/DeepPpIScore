import os
import sys
import MDAnalysis as mda
from functools import partial 
from multiprocessing import Pool
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData
from tqdm import tqdm
import sys
sys.path.append('../')
from utils.fns import load_graph, save_graph


# added by jdj; 20230920
from dataset.protein_feature import get_pocket_feature_mda
from dataset.ligand_feature import  get_pep_node_feat, get_pep_corrd_feat
import joblib
import tempfile

def generate_pocket_graph(pocket_pdb_file, pocket_dst_file, gvp_top_k=30, remove_H=True):
    if not os.path.exists(pocket_dst_file):
        if remove_H:
            pocket_mol = mda.Universe(pocket_pdb_file)
            tempdir = tempfile.mkdtemp(prefix='pepbdbgraph')
            non_h_atoms = pocket_mol.select_atoms("not type H")
            non_h_atoms.write('%s/%s' % (tempdir, os.path.basename(pocket_pdb_file)))
            pocket_mol = mda.Universe('%s/%s' % (tempdir, os.path.basename(pocket_pdb_file)))
            # 删除临时文件
            os.remove('%s/%s' % (tempdir, os.path.basename(pocket_pdb_file)))
            os.rmdir(tempdir)
        else:
            pocket_mol = mda.Universe(pocket_pdb_file)


        # get pocket feats
        p_X_ca, p_X_cb, p_xyz_full, p_seq, p_node_s, p_node_v, p_edge_index, p_edge_s, p_edge_v, p_full_edge_s, p_X_center_of_mass, p_side_chain_mass = get_pocket_feature_mda(pocket_mol, top_k=gvp_top_k)
        data = HeteroData()
        # protein, node feature
        data['protein'].node_s = p_node_s.to(torch.float32) 
        data['protein'].node_v = p_node_v.to(torch.float32)
        # protein, coordinates
        data['protein'].X_ca = p_X_ca.to(torch.float32)
        data['protein'].X_cb = p_X_cb.to(torch.float32)
        data['protein'].X_center_of_mass = p_X_center_of_mass.to(torch.float32)
        data['protein'].side_chain_mass = p_side_chain_mass.to(torch.float32)
        # protein, others
        data['protein'].xyz_full = p_xyz_full.to(torch.float32) 
        data['protein'].seq = p_seq.to(torch.int32)
        data['protein', 'p2p', 'protein'].edge_index = p_edge_index.to(torch.long)
        data['protein', 'p2p', 'protein'].edge_s = p_edge_s.to(torch.float32) 
        # data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
        data['protein', 'p2p', 'protein'].edge_v = p_edge_v.to(torch.float32) 
        save_graph(pocket_dst_file, data)
        return data
    else:
        return load_graph(pocket_dst_file)


def get_pep_corr_graph(pep_pdb_file, pdb_id, ligand_corrd_file, remove_H=True):
    if not os.path.exists(ligand_corrd_file):
        if remove_H:
            pep_mol = mda.Universe(pep_pdb_file)
            tempdir = tempfile.mkdtemp(prefix='pepbdbgraph')
            non_h_atoms = pep_mol.select_atoms("not type H")
            non_h_atoms.write('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            pep_mol = mda.Universe('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            # 删除临时文件
            os.remove('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            os.rmdir(tempdir)
        else:
            pep_mol = mda.Universe(pep_pdb_file)
        pep_X_ca, pep_X_cb, pep_X_center_of_mass, pep_side_chain_mass, pure_res_lis, pep_xyz_full, torsion, pro_rep_node_s, pro_rep_node_v = get_pep_corrd_feat(pep_mol, pdb_id)
        data = HeteroData()
        # peptide, coordinates
        data['peptide'].X_ca = pep_X_ca.to(torch.float32)
        data['peptide'].X_cb = pep_X_cb.to(torch.float32)
        data['peptide'].torsion = torsion.to(torch.float32)
        data['peptide'].X_center_of_mass = pep_X_center_of_mass.to(torch.float32)
        data['peptide'].side_chain_mass = pep_side_chain_mass.to(torch.float32)
        data['peptide'].xyz_full = pep_xyz_full.to(torch.float32)
        data['peptide'].pro_rep_node_s = pro_rep_node_s.to(torch.float32)
        data['peptide'].pro_rep_node_v = pro_rep_node_v.to(torch.float32)

        save_graph(ligand_corrd_file, data)
        return data
    else:
        return load_graph(ligand_corrd_file)


def get_pep_node_s_graph(pep_pdb_file, pdb_id, ligand_node_s_file, remove_H=True):
    if not os.path.exists(ligand_node_s_file):
        if remove_H:
            pep_mol = mda.Universe(pep_pdb_file)
            tempdir = tempfile.mkdtemp(prefix='pepbdbgraph')
            non_h_atoms = pep_mol.select_atoms("not type H")
            non_h_atoms.write('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            pep_mol = mda.Universe('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            # 删除临时文件
            os.remove('%s/%s' % (tempdir, os.path.basename(pep_pdb_file)))
            os.rmdir(tempdir)
        else:
            pep_mol = mda.Universe(pep_pdb_file)
        node_s = get_pep_node_feat(pep_mol, pdb_id)
        data = HeteroData()
        # peptide, coordinates
        data['peptide'].node_s = node_s.to(torch.float32)
        save_graph(ligand_node_s_file, data)
        return data
    else:
        return load_graph(ligand_node_s_file)


class NegPepBDBGraphDataset(Dataset):
    def __init__(self, pdb_ids, pocket_files, ligand_files, global_feats, args, pki_labels=None, dataset_type='train'):
        '''
        :param pdb_ids: pdb_ids
        :param pocket_files: pocket files
        :param ligand_files: ligand files
        :global_feats, list, 针对于每个多肽构象的global_feature, ['Fnat', 'Fnonnat', 'iRMS', 'LRMS', 'DockQ', 'glob', 'aVdW','rVdW', 'ACE', 'inside', 'aElec', 'rElec', 'laElec', 'lrElec', 'hb', 'piS', 'catpiS', 'aliph', 'affinity', 'ref_backbone_rmsd', 'ref_fnc']
        :param pki_labels: pki/pkd/ic50 of protein-ligand complexes
        :param dataset_type: in ['train', 'valid', 'test']
        :param n_job: if n_job == 1: use for-loop;else: use multiprocessing
        :param on_the_fly: whether to get graph from a totoal graph list or a single graph file
        :param graph_gen: what graph generation function should be used
        _______________________________________________________________________________________________________
        |  mode  |  generate single graph file  |  generate integrated graph file  |  load to memory at once  |
        |  False |          No                  |              Yes                 |            Yes           |
        |  True  |          Yes                 |              No                  |            No            |
        |  Fake  |          Yes                 |              No                  |            Yes           |
        _______________________________________________________________________________________________________
        '''
        self.pdb_ids = pdb_ids
        self.pocket_files = pocket_files
        self.ligand_files = ligand_files
        self.pocket_dst_dir = args["pocket_dst_dir"]
        self.ligand_dst_dir = args["ligand_dst_dir"]
        self.global_feats = global_feats
        if pki_labels is not None:
            self.pki_labels = pki_labels
        else:
            self.pki_labels = np.zeros((len(self.pdb_ids)))
        os.makedirs(self.pocket_dst_dir, exist_ok=True)
        os.makedirs(self.ligand_dst_dir, exist_ok=True)
        assert dataset_type in ['train', 'valid', 'test'], 'illegal dataset type'
        assert (len(pdb_ids) == len(pocket_files) == len(ligand_files) == len(global_feats)), 'the length of pdb_ids, pocket_files, ligand_files and global_feats should be equal'
        self.dataset_type = dataset_type
        self.dst_file = f'{self.ligand_dst_dir}/{dataset_type}.dgl'
        self.n_job = args['num_cpu']
        self.on_the_fly = args['on_the_fly']
        assert self.on_the_fly in [True, False, 'Fake']
        self.verbose = args["verbose"]
        
        self.graph_gen = args["graph_gen"]
        self.graph_labels = []
        self.add_inter_g = args["add_inter_g"]
        self.gvp_top_k = args["gvp_top_k"]
        self.remove_H = args["remove_H"]

        self.pre_process()

    def pre_process(self):
        if self.on_the_fly == 'Fake':
            self._generate_graph_on_the_fly_fake()
        elif self.on_the_fly:
            self._generate_graph_on_the_fly()
        else:
            self._generate_graph()

    def _generate_graph(self):
        if os.path.exists(self.dst_file):
            if self.verbose:
                print('load graph')
            self.graph_labels = load_graph(self.dst_file)
        else:
            idxs = range(len(self.pdb_ids))
            if self.verbose:
                print('### cal graph')
            single_process = partial(self._single_process, return_graph=True, save_file=False)
            # generate graph
            if self.n_job == 1:
                if self.verbose:
                    idxs = tqdm(idxs)
                for idx in idxs:
                    self.graph_labels.append(single_process(idx))
            else:
                pool = Pool(self.n_job)
                self.graph_labels = pool.map(single_process, idxs)
                pool.close()
                pool.join()
            # filter None
            self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))
            # save file
            save_graph(self.dst_file, self.graph_labels)

    def _generate_graph_on_the_fly(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly')
        single_process = partial(self._single_process, return_graph=False, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                single_process(idx)
        else:
            print('### processing graph with %s cpu' % self.n_job)
            pool = Pool(self.n_job)
            pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # self.pdb_ids = [os.path.split(i)[-1].split('.')[0] for i in glob.glob(f'{self.dst_dir}/*.dgl')]

    def _generate_graph_on_the_fly_fake(self):
        idxs = range(len(self.pdb_ids))
        if self.verbose:
            print('### get graph on the fly (fake)')
        single_process = partial(self._single_process, return_graph=True, save_file=True)
        # generate graph
        if self.n_job == 1:
            if self.verbose:
                idxs = tqdm(idxs)
            for idx in idxs:
                self.graph_labels.append(single_process(idx))
        else:
            pool = Pool(self.n_job)
            self.graph_labels = pool.map(single_process, idxs)
            pool.close()
            pool.join()
        # filter None
        self.graph_labels = list(filter(lambda x: x is not None, self.graph_labels))
    
    def _make_pair_graph(self, pocket_graph=None, ligand_coord_graph=None, ligand_node_s_graph=None, ligand_graph=None):
        data = HeteroData()
        if self.graph_gen == 'v1':
            # protein, node feature
            data['protein'].node_s = pocket_graph['protein'].node_s
            data['protein'].node_v = pocket_graph['protein'].node_v
            # protein, coordinates
            data['protein'].X_ca = pocket_graph['protein'].X_ca
            data['protein'].X_cb = pocket_graph['protein'].X_cb
            data['protein'].X_center_of_mass = pocket_graph['protein'].X_center_of_mass
            data['protein'].side_chain_mass = pocket_graph['protein'].side_chain_mass
            # protein, others
            data['protein'].xyz_full = pocket_graph['protein'].xyz_full
            data['protein'].seq = pocket_graph['protein'].seq
            data['protein', 'p2p', 'protein'].edge_index = pocket_graph['protein', 'p2p', 'protein'].edge_index
            data['protein', 'p2p', 'protein'].edge_s = pocket_graph['protein', 'p2p', 'protein'].edge_s
            # data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
            data['protein', 'p2p', 'protein'].edge_v = pocket_graph['protein', 'p2p', 'protein'].edge_v

            # peptide,feature
            # pro_rep_node_s, [num_res, 9]
            data['peptide'].node_s = ligand_node_s_graph['peptide'].node_s
            # data['peptide'].node_s = torch.cat([ligand_node_s_graph['peptide'].node_s[:, :678], ligand_coord_graph['peptide'].pro_rep_node_s, ligand_coord_graph['peptide'].pro_rep_node_v.view(-1, 9)], dim=1)
            
            # peptide, coordinates
            data['peptide'].X_ca = ligand_coord_graph['peptide'].X_ca
            data['peptide'].X_cb = ligand_coord_graph['peptide'].X_cb
            data['peptide'].torsion = ligand_coord_graph['peptide'].torsion 
            data['peptide'].X_center_of_mass = ligand_coord_graph['peptide'].X_center_of_mass
            data['peptide'].side_chain_mass = ligand_coord_graph['peptide'].side_chain_mass
            data['peptide'].xyz_full = ligand_coord_graph['peptide'].xyz_full

            if self.add_inter_g:
                 edge_index, dists = get_inter_g_edge_index(data=data)
                 data['inter_g'].node_s = edge_index  # [num_edges, 2]
                 data['inter_g'].dists = dists

        elif self.graph_gen == 'v2':
            # protein, node feature
            data['protein'].node_s = pocket_graph['protein'].node_s
            data['protein'].node_v = pocket_graph['protein'].node_v
            # protein, coordinates
            data['protein'].X_ca = pocket_graph['protein'].X_ca
            data['protein'].X_cb = pocket_graph['protein'].X_cb
            data['protein'].X_center_of_mass = pocket_graph['protein'].X_center_of_mass
            data['protein'].side_chain_mass = pocket_graph['protein'].side_chain_mass
            # protein, others
            data['protein'].xyz_full = pocket_graph['protein'].xyz_full
            data['protein'].seq = pocket_graph['protein'].seq
            data['protein', 'p2p', 'protein'].edge_index = pocket_graph['protein', 'p2p', 'protein'].edge_index
            data['protein', 'p2p', 'protein'].edge_s = pocket_graph['protein', 'p2p', 'protein'].edge_s
            # data['protein', 'p2p', 'protein'].full_edge_s = p_full_edge_s.to(torch.float32) 
            data['protein', 'p2p', 'protein'].edge_v = pocket_graph['protein', 'p2p', 'protein'].edge_v
            data['peptide'].node_s = ligand_graph['peptide'].node_s
            data['peptide'].node_v = ligand_graph['peptide'].node_v
            # peptide, coordinates
            data['peptide'].X_ca = ligand_graph['peptide'].X_ca
            data['peptide'].X_cb = ligand_graph['peptide'].X_cb
            data['peptide'].X_center_of_mass = ligand_graph['peptide'].X_center_of_mass
            data['peptide'].side_chain_mass = ligand_graph['peptide'].side_chain_mass
            # peptide, others
            data['peptide'].xyz_full = ligand_graph['peptide'].xyz_full
            data['peptide'].seq = ligand_graph['peptide'].seq
            data['peptide', 'p2p', 'peptide'].edge_index = ligand_graph['peptide', 'p2p', 'peptide'].edge_index
            data['peptide', 'p2p', 'peptide'].edge_s = ligand_graph['peptide', 'p2p', 'peptide'].edge_s
            # data['peptide', 'p2p', 'peptide'].full_edge_s = p_full_edge_s.to(torch.float32) 
            data['peptide', 'p2p', 'peptide'].edge_v = ligand_graph['peptide', 'p2p', 'peptide'].edge_v
        
        elif self.graph_gen == 'v3':
            data['protein'].pos = pocket_graph['protein'].pos.to(torch.float32) 
            data['protein'].node_feature = pocket_graph['protein'].node_feature.to(torch.float32)
            data['protein'].edge_feature = pocket_graph['protein'].edge_feature.to(torch.float32)
            data['protein'].edge_index = pocket_graph['protein'].edge_index.to(torch.long)

            # peptide
            data['peptide'].pos = ligand_graph['peptide'].pos.to(torch.float32) 
            data['peptide'].node_feature = ligand_graph['peptide'].node_feature.to(torch.float32)
            data['peptide'].edge_feature = ligand_graph['peptide'].edge_feature.to(torch.float32)
            data['peptide'].edge_index = ligand_graph['peptide'].edge_index.to(torch.long)

        return data

    def _single_process(self, idx, return_graph=False, save_file=False):
        pdb_id = self.pdb_ids[idx]
        pocket_pdb_file = self.pocket_files[idx]
        pep_pdb_file = self.ligand_files[idx]
        global_feature = self.global_feats[idx]
        # graph_name = pdb_id + '_' + os.path.basename(pep_pdb_file).split('.')[0]
        # pocket_name = os.path.basename(pocket_pdb_file).split('.')[0]
        pocket_name = os.path.basename(pocket_pdb_file)[:-4] # 文件以.pdb结尾
        # ligand_name = os.path.basename(pep_pdb_file).split('.')[0]
        ligand_name = os.path.basename(pep_pdb_file)[:-4]  # 文件以.pdb结尾
        pocket_dst_file = f'{self.pocket_dst_dir}/{pdb_id}_{pocket_name}.dgl'
        ligand_graph_file = f'{self.ligand_dst_dir}/{pdb_id}_{ligand_name}.dgl'
        ligand_corrd_file = f'{self.ligand_dst_dir}/{pdb_id}_{ligand_name}.dgl'  # -> 存储坐标
        # # ligand_corrd_file重新生成
        # ligand_coord_graph = get_pep_corr_graph(pep_pdb_file, pdb_id, ligand_corrd_file, remove_H=self.remove_H)
        # save_graph(ligand_corrd_file, ligand_coord_graph)
        ligand_node_s_file = f'{self.ligand_dst_dir}/{pdb_id}_pep_node_s.dgl'  # -> 针对于同一个多肽的不同构象，其node_s是一样的

        if os.path.exists(pocket_dst_file) and os.path.exists(ligand_corrd_file) and os.path.exists(ligand_node_s_file):
            # reload graph
            if return_graph:
                if self.graph_gen == 'v1':
                    pocket_graph, ligand_coord_graph, ligand_node_s_graph = load_graph(pocket_dst_file), load_graph(ligand_corrd_file), load_graph(ligand_node_s_file)
                    data = self._make_pair_graph(pocket_graph, ligand_coord_graph, ligand_node_s_graph)
                    data.pdb_id = pdb_id
                    data.ligand_name = ligand_name
                    data.global_feature = global_feature
                    return data
                elif self.graph_gen == 'v2':
                    pocket_graph, ligand_graph = load_graph(pocket_dst_file), load_graph(ligand_graph_file)
                    data = self._make_pair_graph(pocket_graph=pocket_graph, ligand_coord_graph=None, ligand_node_s_graph=None, ligand_graph=ligand_graph)
                    data.pdb_id = pdb_id
                    data.ligand_name = ligand_name
                    data.global_feature = global_feature
                    return data

        else:
            # generate graph
            pki_label = self.pki_labels[idx]
            try:
                if self.graph_gen == 'v1':
                    if not os.path.exists(pocket_dst_file):
                        pocket_graph = generate_pocket_graph(pocket_pdb_file, pocket_dst_file, gvp_top_k=self.gvp_top_k, remove_H=self.remove_H)
                        # save_graph(pocket_dst_file, pocket_graph)
                    if not os.path.exists(ligand_corrd_file):
                        ligand_coord_graph = get_pep_corr_graph(pep_pdb_file, pdb_id, ligand_corrd_file, remove_H=self.remove_H)
                        # save_graph(ligand_corrd_file, ligand_coord_graph)
                    if not os.path.exists(ligand_node_s_file):
                        ligand_node_s_graph = get_pep_node_s_graph(pep_pdb_file, pdb_id, ligand_node_s_file, remove_H=self.remove_H)
                        # save_graph(ligand_node_s_file, ligand_node_s_graph)
                    # pocket_graph, ligand_coord_graph, ligand_node_s_graph = generate_pepset_pocket_graph(pocket_pdb_file, pocket_dst_file, gvp_top_k=self.gvp_top_k, remove_H=self.remove_H), get_pep_corr_graph(pep_pdb_file, pdb_id, ligand_corrd_file, remove_H=self.remove_H), get_pep_node_s_graph(pep_pdb_file, pdb_id, ligand_node_s_file, remove_H=self.remove_H)
                elif self.graph_gen == 'v2':
                    pocket_graph, ligand_graph = generate_graph_v2_2(pocket_pdb_file, pep_pdb_file)
                    if not os.path.exists(ligand_graph_file):
                        save_graph(ligand_graph_file, ligand_graph)
                elif self.graph_gen == 'v3':
                    pocket_graph, ligand_graph = generate_graph_v3_2(pocket_pdb_file, pep_pdb_file)

                if return_graph:
                    if self.graph_gen == 'v1':
                        # 保证多肽的坐标图数据和node_s的结点数量一致
                        if (ligand_node_s_graph['peptide'].node_s.shape[0] == ligand_coord_graph['peptide'].X_ca.shape[0] == ligand_coord_graph['peptide'].xyz_full.shape[0]):
                            data = self._make_pair_graph(pocket_graph, ligand_coord_graph, ligand_node_s_graph)
                            data.pdb_id = pdb_id
                            data.ligand_name = ligand_name
                            data.global_feature = global_feature
                            return data
                        else:
                            print(f'{pep_pdb_file} inconsistent peptide node error')
                            return None
                    elif self.graph_gen == 'v2':
                        pocket_graph, ligand_graph = load_graph(pocket_dst_file), load_graph(ligand_graph_file)
                        data = self._make_pair_graph(pocket_graph=pocket_graph, ligand_coord_graph=None, ligand_node_s_graph=None, ligand_graph=ligand_graph)
                        data.pdb_id = pdb_id
                        data.ligand_name = ligand_name
                        data.global_feature = global_feature
                        return data
            except:
                print(f'{pep_pdb_file} error')
                return None

    def __getitem__(self, idx):
        if self.on_the_fly == True:
            data = self._single_process(idx=idx, return_graph=True, save_file=False)
        else:
            data = self.graph_labels[idx]
        return data


    def __len__(self):
        if self.on_the_fly == True:
            return len(self.pdb_ids)
        else:
            return len(self.graph_labels)

   
# # 根据残基-残基的cb-cb距离构建每个datapoint所对应的二分图的边
def get_inter_g_edge_index(data, distance_cutoff=18.0): 
    # data should be single data point, not batch data point
    '''
    边的方向: protein node -> peptide node
    边的方向: h_l_pos -> h_t_pos 
    '''   

    # h_l_pos = data['protein'].xyz_full
    # h_t_pos = data['peptide'].xyz_full

    # # 假定 h_l_pos 和 h_t_pos为给定的tensor，形状分别为 [7, 24, 3] 和 [182, 24, 3] 
    # # 计算两个tensor每个点之间的差的平方
    # # 使用unsqueeze来扩展维度，以便广播
    # # h_l_pos_expanded 形状为 [7, 1, 24, 3]
    # # h_t_pos_expanded 形状为 [1, 182, 24, 3]
    # h_l_pos_expanded = h_l_pos.unsqueeze(1).unsqueeze(3)  # -> [7, 1, 24, 1, 3]
    # h_t_pos_expanded = h_t_pos.unsqueeze(0).unsqueeze(2)  # -> [1, 182, 1, 24, 3]

    # # 广播计算所有原子对之间的距离平方
    # # 结果形状将会是 [7, 182, 24, 24]
    # dist_sq = (h_l_pos_expanded - h_t_pos_expanded) ** 2

    # # 距离平方和，结果形状为 [7, 182, 24, 24]
    # dist_sq_sum = torch.sum(dist_sq, dim=-1)
    # del dist_sq
    # gc.collect()
    # torch.cuda.empty_cache()

    # # 将 NaN 替换为正无穷大，这样它们就不会影响最小值的计算
    # dist_sq_sum = torch.nan_to_num(dist_sq_sum, nan=float('inf'))

    # # 沿着原子维度（dim=2和dim=3）找到最小距离平方
    # # 我们首先需要处理第一个残基的所有原子与第二个残基的所有原子之间的距离
    # min_dist_sq, _ = torch.min(dist_sq_sum, dim=3)  # 最小距离平方，形状为 [7, 182, 24]
    # min_dist_sq, _ = torch.min(min_dist_sq, dim=2)  # 最小距离平方，形状为 [7, 182]

    # # 计算实际的最小距离
    # min_dist = torch.sqrt(min_dist_sq)

    # # src_index is from protein node
    # # dst_index is from peptide node
    # src_index, dst_index = torch.where(min_dist < distance_cutoff)[0], torch.where(min_dist < distance_cutoff)[1]
    # # 返回min-min距离值
    # dist1 = min_dist[torch.where(min_dist < distance_cutoff)].view(-1,1)

    # x_ca_dist = torch.cdist(data['protein'].X_ca, data['peptide'].X_ca)
    # src_index, dst_index = torch.where(x_cb_dist < distance_cutoff)[0], torch.where(x_cb_dist < distance_cutoff)[1]
    # dist1 = x_cb_dist[torch.where(x_cb_dist < distance_cutoff)].view(-1,1)


    # ca-ca距离
    x_ca_dist = torch.cdist(data['protein'].X_ca, data['peptide'].X_ca)
    src_index, dst_index = torch.where(x_ca_dist < distance_cutoff)[0], torch.where(x_ca_dist < distance_cutoff)[1]
    dist2 = x_ca_dist[src_index, dst_index].view(-1, 1)
    # # side-side距离
    # side_dist = torch.cdist(data['protein'].side_chain_mass, data['peptide'].side_chain_mass)
    # dist3 = side_dist[src_index, dst_index].view(-1, 1)
    
    # dists = torch.cat((dist1, dist2, dist3), dim=1)

    # dst_index = dst_index + h_l_pos.size()[0]

    # edge_index = torch.stack([src_index, dst_index])
    edge_index = torch.stack([src_index, dst_index], dim=1)  # [num_edges, 2]

    # data['inter_g'].node_s = edge_index  # [num_edges, 2]
    # data['inter_g'].dists = dists
    return edge_index.to(torch.long), dist2.to(torch.float32)