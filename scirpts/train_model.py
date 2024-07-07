# 计算所有的距离只考虑重原子，残基的最大原子数量限制在15。
# 蛋白多肽均采用残基表征，蛋白口袋用gvp, 多肽用基于序列的transformer
# 分别拟合二种距离, cb-cb (18A); minimum-minimum (12A)
# 加入adcp对接的high quality构象进行训练
import sys
sys.path.append('../')
from dataset.graph_obj import NegPepBDBGraphDataset
from architecture.DeepPpIScore import DeepPpIScore
from utils.fns import run_a_train_epoch, run_an_eval_epoch, Early_stopper, set_random_seed
import torch 
import numpy as np
from dataset.dataloader_obj import PassNoneDataLoader
from prefetch_generator import BackgroundGenerator
import os
import copy
import time
import multiprocessing as mp
import pandas as pd
import pickle
import joblib
class DataLoaderX(PassNoneDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


args={}
args["test_code"] = False
args["save_model_per_epoch"] = False
args["model_tag"] = "DeepPpIScore_test"  # cb-cb (18A); minimum-minimum (12A)

## 模型文件保存位置
args["model_path"] = "../trained_models/%s.pth" % args["model_tag"]


## 图生成参数
args["add_inter_g"] = False # 是否在每个paired data point中添加相互作用图
args["verbose"] = False 
args["gvp_top_k"] = 30  # gvp图数据的top-k
args["remove_H"] = True # 是否从读入的pdb文件中删除H
args['on_the_fly']=True
args["graph_gen"] = 'v1'  # 'v4'表示使用receptor_inter_chains.pdb做为蛋白受体
args['pocket_dst_dir'] = '../data/pepbdb_graphs_noH_pocket_topk%s' % (args["gvp_top_k"])    
args['ligand_dst_dir'] = '../data/pepbdb_graphs_noH_ligand'   


## 模型参数
args["n_gaussians"] = 10
args["node_embed_size"] = 128
args["pep_in_feat_size"] = 1957+1 # 多肽残基的初始特征维度，多一维多肽内部残基的编号
args["pep_encoder_n_head"] = 8 # 多肽transformer n_head
args["pep_encoder_n_layer"] = 6 # 多肽transformer layer
args["pro_node_in_dim"] = (9, 3)  # 蛋白输入节点的维度，（标量特征维度，向量特征维度）
args["pro_node_h_dim"] = (args["node_embed_size"], 16)  # 蛋白边的维度，（标量特征维度，向量特征维度）
args["pro_edge_in_dim"] = (102, 1)  # 蛋白边的维度，（标量特征维度，向量特征维度）
args["pro_edge_h_dim"] = (32, 1)  # 蛋白边的维度，（标量特征维度，向量特征维度）
args["pro_num_layers"] = 3 
args["pro_seq_in"] = True  

## 训练参数
args["batch_size"] = 32
args["dropout_rate"] = 0.2
args['patience'] = 70
args['tolerance'] = 0.005
args["num_workers"] = 4
args['mode'] = "lower"
args['lr'] = 3
args['weight_decay'] = 5
args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args['seeds'] = 43 
args["dist_threhold1"] = 18.0  # cb-cb训练阈值
args["dist_threhold2"] = 12.0  # minimum-minimum训练阈值
args["val_fra"] = 0.2
args["num_epochs"] = 5000
args["num_cpu"] = mp.cpu_count()

print('the current arguments: \n', args)


training_pose = joblib.load('../data/training_pose_.dgl')
pdb_ids = training_pose['pdb_ids']
pocket_files = training_pose['pocket_files']
ligand_files = training_pose['ligand_files']
global_feats = training_pose['global_feats']
pocket_dst_files  = training_pose['pocket_dst_files']
ligand_node_s_files = training_pose['ligand_node_s_files']
ligand_corrd_files = training_pose['ligand_corrd_files']

print('training dataset...')
print('exmaple of files is:')
print('pdb_ids: %s' % pdb_ids[:3])
print('pocket_dst_files: %s' % pocket_dst_files[:3])
print('ligand_node_s_files: %s' % ligand_node_s_files[:3])
print('ligand_corrd_files: %s' % ligand_corrd_files[:3])
print('the total number of training pose is: %s' % len(ligand_files))
train_dataset = NegPepBDBGraphDataset(pdb_ids=pdb_ids, pocket_files=pocket_files, ligand_files=ligand_files, global_feats=global_feats, args=args)
print('training dataset...')

validation_pose = joblib.load('../data/validation_pose_.dgl')
pdb_ids = validation_pose['pdb_ids']
pocket_files = validation_pose['pocket_files']
ligand_files = validation_pose['ligand_files']
global_feats = validation_pose['global_feats']
pocket_dst_files  = validation_pose['pocket_dst_files']
ligand_node_s_files = validation_pose['ligand_node_s_files']
ligand_corrd_files = validation_pose['ligand_corrd_files']

print('validation dataset...')
print('exmaple of files is:')
print('pdb_ids: %s' % pdb_ids[:3])
print('pocket_dst_files: %s' % pocket_dst_files[:3])
print('ligand_node_s_files: %s' % ligand_node_s_files[:3])
print('ligand_corrd_files: %s' % ligand_corrd_files[:3])
print('the total number of validation pose is: %s' % len(ligand_files))
valid_dataset = NegPepBDBGraphDataset(pdb_ids=pdb_ids, pocket_files=pocket_files, ligand_files=ligand_files, global_feats=global_feats, args=args)
print('validation dataset...')


model = DeepPpIScore(args=args)
model = model.to(args['device'])
optimizer = torch.optim.Adam(model.parameters(), lr=10**-args['lr'], weight_decay=10**-args['weight_decay'])

# dataloader
train_dataloader = DataLoaderX(dataset=train_dataset,  batch_size=args['batch_size'], shuffle=True, num_workers=args["num_workers"], follow_batch=[], pin_memory=True)
valid_dataloader = DataLoaderX(dataset=valid_dataset,  batch_size=args['batch_size'], shuffle=True, num_workers=args["num_workers"], follow_batch=[], pin_memory=True)
stopper = Early_stopper(patience=args['patience'], mode=args['mode'], model_file=args["model_path"])


set_random_seed(args["seeds"])
for epoch in range(args["num_epochs"]):
    st = time.time()	
    # Train, 
    total_loss_train = run_a_train_epoch(model, train_dataloader, optimizer, dist_threhold1 = args['dist_threhold1'], 
                                             dist_threhold2 = args['dist_threhold2'], device=args["device"])	
    if args["save_model_per_epoch"]:
        stopper.save_model_per_epoch(model, epoch)
    if np.isinf(total_loss_train) or np.isnan(total_loss_train): 
        print('Inf ERROR')
        break
    # validation and early stop
    total_gm_loss, gm_loss1, gm_loss2 = run_an_eval_epoch(model, valid_dataloader,dist_threhold1 = args['dist_threhold1'], 
                                             dist_threhold2 = args['dist_threhold2'], device=args["device"])
    early_stop = stopper.step(total_gm_loss, model)
    end = time.time()
    print("epoch:%s, b_val:%.3f,time:%.2fS,tl:%.3f,loss1:%.3f,loss2:%.3f" % (epoch + 1, stopper.best_score, end - st, total_gm_loss, gm_loss1, gm_loss2))
    if early_stop:
        break

stopper.load_model(model, args["device"])
total_gm_loss, gm_loss1, gm_loss2 = run_an_eval_epoch(model, train_dataloader, dist_threhold1 = args['dist_threhold1'], 
                                             dist_threhold2 = args['dist_threhold2'],  device=args["device"])
print("train set: tl:%.3f,loss1:%.3f,loss2:%.3f" % (total_gm_loss, gm_loss1, gm_loss2))

total_gm_loss, gm_loss1, gm_loss2 = run_an_eval_epoch(model, valid_dataloader, dist_threhold1 = args['dist_threhold1'], 
                                             dist_threhold2 = args['dist_threhold2'],  device=args["device"])		
print("valid set: tl:%.3f,loss1:%.3f,loss2:%.3f" % (total_gm_loss, gm_loss1, gm_loss2))

