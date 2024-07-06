import sys
sys.path.append('../')
from dataset.graph_obj import NegPepBDBGraphDataset
from architecture.DeepPpIScore import DeepPpIScore
from utils.fns import run_an_eval_epoch
import torch 
from dataset.dataloader_obj import PassNoneDataLoader
from prefetch_generator import BackgroundGenerator
import os
import copy
import time
import multiprocessing as mp
import pandas as pd
import re
from torch_scatter import scatter_add, scatter_sum
import glob
class DataLoaderX(PassNoneDataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

#########################################################
## assign input files here, including  pocket_files, ligand_files, pdb_ids
pocket_files, ligand_files, pdb_ids, global_feats  = [], [], [], []
pattern = re.compile(r'model_\d+\.pdb$')  # 配体文件的正则表达式
pocket_file = '../inference_input_example/3hds/3hds_rec_unbound_pocket_20.0.pdb'
files = os.listdir('../inference_input_example/3hds')
for file in files:
    if pattern.match(file):
        ligand_files.append('../inference_input_example/3hds/%s' %  file)
        pocket_files.append(pocket_file)
        pdb_ids.append('3hds')
        global_feats.append([0])
## assign input files here, including pocket_files, ligand_files, pdb_ids
#########################################################


#########################################################
## evaluation setting
args={}
args["test_code"] = False
args["eval_on_gpu"] = True
args["graph_gen_on_cpu"] = True
args["model_tag"] = "DeepPpIScore"

args["model_path"] = "../trained_models/%s.pth" % args["model_tag"]
args["csv_out_path"] = "../model_inference/%s" % args["model_tag"]
args["graph_file_save_path"] = "../data/temp_graphs_noH"
args["on_the_fly"] = True
args["graph_gen"] = 'v1' 
args["num_cpu"] = mp.cpu_count()
args["verbose"] = False
args["add_inter_g"] = False
args["gvp_top_k"] = 30
args["remove_H"] = True
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
args["dropout_rate"] = 0.2
args['seeds'] = 43
args["dist_threhold1"] = 18.0  # cb-cb训练阈值
args["dist_threhold2"] = 12.0  # minimum-minimum训练阈值
args["num_cpu"] = mp.cpu_count()
args["batch_size"] = 64
args["num_workers"] = 10
args['device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
args["dist_threhold_eval_ls"] = [8.0]  
if args['test_code']:
    args["limit"] = 10
else:
    args["limit"] = None
ligand_dst_dir = '%s/%s' % (args["graph_file_save_path"],  args['graph_gen'])
pocket_dts_dir = '%s/shared_pocket_topk%s' % (args["graph_file_save_path"], args["gvp_top_k"])
args["ligand_dst_dir"] = ligand_dst_dir
args["pocket_dst_dir"] = pocket_dts_dir
if not os.path.exists(args["csv_out_path"]):
    cmdline = 'mkdir -p %s' % args["csv_out_path"]
    os.system(cmdline)
if not os.path.exists(args["ligand_dst_dir"]):
    cmdline = 'mkdir -p %s' % args["ligand_dst_dir"]
    os.system(cmdline)
if not os.path.exists(args["pocket_dst_dir"]):
    cmdline = 'mkdir -p %s' % args["pocket_dst_dir"]
    os.system(cmdline)
## evaluation setting
#########################################################


print('the current arguments', args)

print('graph data generation is starting on %s...' % args["device"])
st_time = time.time()
print('the number of ligand files and pocket_files are: %s, %s, respectively' % (len(ligand_files), len(pocket_files)))
print('the example of ligand files are: %s' % (ligand_files[:10]))
print('the example of pocket_files are: %s' % (pocket_files[:10]))
dataset = NegPepBDBGraphDataset(pdb_ids=pdb_ids[:args["limit"]], pocket_files=pocket_files[:args["limit"]], ligand_files=ligand_files[:args["limit"]], 
                            global_feats=global_feats[:args["limit"]], args=args)
end_time = time.time()
print('graph data generation is ending on %s..., elapsed time: %s S' % (args["device"], end_time - st_time))

if args["eval_on_gpu"] and args["device"] == 'cpu':
    print("no gpu device for evaluation, skip evaluation...")
else:
    print('evaluation is starting on %s...' % args["device"])

    st_time = time.time()
    model = DeepPpIScore(args=args)
    checkpoint = torch.load(args["model_path"], map_location=torch.device(args['device']))
    model.load_state_dict(checkpoint['model_state_dict']) 
    print('model loading is okay...')
    model = model.to(args['device'])
    dataloader = DataLoaderX(dataset=dataset,  batch_size=args['batch_size'], shuffle=False, num_workers=args["num_workers"], follow_batch=[], pin_memory=True)
    pdbids, ligand_names, probs_ls1, batch_ls1,  dist_ls1, probs_ls2, batch_ls2,  dist_ls2, batch_size_ls = run_an_eval_epoch(model, dataloader, dist_threhold1 = args['dist_threhold1'], dist_threhold2 = args['dist_threhold2'], device=args["device"], pred=True)
    for threhold in args["dist_threhold_eval_ls"]:
        csv_out_file = '%s/%s/%s_%s.csv' % (args["csv_out_path"], threhold, args["model_tag"], threhold)
        if not os.path.exists(os.path.dirname(csv_out_file)):
            cmdline = 'mkdir -p %s' % os.path.dirname(csv_out_file)
            os.system(cmdline)
        probs, norm_probs, num_contacts = [], [], []
        for prob1, batch1, dist1, prob2, batch2, dist2, batch_size in zip(probs_ls1, batch_ls1,  dist_ls1, probs_ls2, batch_ls2,  dist_ls2, batch_size_ls):
            prob1_cp = copy.deepcopy(prob1)
            prob1_cp[torch.where(dist1 > threhold)[0]] = 0.   
            batch1 = batch1.to(args["device"])
            probx1 = scatter_add(prob1_cp, batch1, dim=0, dim_size=batch_size)
            num_atom_contacts1 = scatter_sum((dist1 <= threhold)*1, batch1, dim=0, dim_size=batch_size)  
            norm_probx1 = probx1/(torch.sqrt(num_atom_contacts1.squeeze()) + 1e-6)

            prob2_cp = copy.deepcopy(prob2)
            prob2_cp[torch.where(dist2 > threhold)[0]] = 0.  
            batch2 = batch2.to(args["device"])
            probx2 = scatter_add(prob2_cp, batch2, dim=0, dim_size=batch_size)
            num_atom_contacts2 = scatter_sum((dist2 <= threhold)*1, batch2, dim=0, dim_size=batch_size)
            norm_probx2 = probx2/(torch.sqrt(num_atom_contacts2.squeeze()) + 1e-6)

            probs.append(torch.stack((probx1, probx2), dim=1))
            norm_probs.append(torch.stack((norm_probx1, norm_probx2), dim=1))
            num_contacts.append(torch.cat([num_atom_contacts1, num_atom_contacts2], dim=1))  
            
        preds = torch.cat(probs)
        norm_preds = torch.cat(norm_probs)
        contacts = torch.cat(num_contacts).squeeze()
        preds, norm_preds, contacts  = preds.cpu().detach().numpy(), norm_preds.cpu().detach().numpy(), contacts.cpu().detach().numpy()
        results = pd.DataFrame({'pdbid': pdbids, 'ligand_name': ligand_names, 
                                'cb-cb score':preds[:, 0], 'cb-cb norm score':norm_preds[:, 0], 
                                'min-min score':preds[:, 1], 'min-min norm score':norm_preds[:, 1], 
                                'cb-cb contacts':contacts[:, 0], 'min-min contacts':contacts[:, 1]})
        results.to_csv(csv_out_file, index=False)
