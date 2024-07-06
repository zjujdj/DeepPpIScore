import torch
import random
import numpy as np
from joblib import load, dump


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def save_graph(dst_file, data):
    dump(data, dst_file)


def load_graph(src_file):
    return load(src_file)


class Early_stopper(object):
    def __init__(self, model_file, mode='higher', patience=70, tolerance=0.0):
        self.model_file = model_file
        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower
        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def load_model(self, model_obj, my_device, strict=False):
        '''Load model saved with early stopping.'''
        model_obj.load_state_dict(torch.load(self.model_file, map_location=my_device)['model_state_dict'], strict=strict)

    def save_model(self, model_obj):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file)
    
    def save_model_per_epoch(self, model_obj, epoch):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model_obj.state_dict()}, self.model_file.replace('.pth', '_epoch%s.pth' % epoch))

    def step(self, score, model_obj):
        if self.best_score is None:
            self.best_score = score
            self.save_model(model_obj)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_model(model_obj)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'# EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        print(f'# Current best performance {float(self.best_score):.3f}')
        return self.early_stop



from torch.distributions import Normal, Categorical
from torch_scatter import scatter_add, scatter_sum, scatter_max


def gm_loss_fn_sv(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc)+1e-10)
    # print(loglik.shape)
    loss = -torch.logsumexp(torch.log(pi+1e-10) + loglik + 1e-10, dim=1)
    return loss


def calculate_probablity_sv(pi, sigma, mu, y):
    normal = Normal(mu, sigma)  # 概率密度函数
    logprob = normal.log_prob(y.expand_as(normal.loc)+1e-10)  # 求解概率密度的对数
    logprob += torch.log(pi+1e-10)
    prob = logprob.exp().sum(1)  # 还原成概率密度,概率密度可大于1 
    return prob


def run_a_train_epoch(model, data_loader, optimizer, dist_threhold1=None, dist_threhold2=None, device='cpu'):
    # dist_threhold1 -> for cb-cb
    # dist_threhold2 -> for min-min

    model.train()
    total_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        try:
            data = batch_data.to(device)

            pi1, sigma1, mu1, dist1, batch1, pi2, sigma2, mu2, dist2, batch2 = model(data)

            gm_loss_true1 = gm_loss_fn_sv(pi1, sigma1, mu1, dist1) # cb-cb
            gm_loss_true1 = gm_loss_true1[torch.where(dist1 <= dist_threhold1)[0]].mean().float() 

            gm_loss_true2 = gm_loss_fn_sv(pi2, sigma2, mu2, dist2) # min-min
            gm_loss_true2 = gm_loss_true2[torch.where(dist2 <= dist_threhold2)[0]].mean().float() 

            loss = gm_loss_true1 + gm_loss_true2

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()

            batch_size = data['peptide'].batch.max()+1
            batch_size = batch_size.item()
            total_loss += (gm_loss_true1.item()*batch_size + gm_loss_true2.item()*batch_size)
            
            if np.isinf(total_loss) or np.isnan(total_loss): break

            del data, pi1, sigma1, mu1, dist1, batch1, pi2, sigma2, mu2, dist2, 
            batch2, gm_loss_true1, gm_loss_true2
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                continue
    return total_loss/len(data_loader.dataset)


def run_an_eval_epoch(model, data_loader, dist_threhold1=None, dist_threhold2=None, device='cpu', pred=False):
    model.eval()
    # probs = [] # batch_size, 3; ca-ca; side-side; minimum-minimum
    # norm_probs  = [] # batch_size, 3
    # num_contacts = [] # batch_size, 3
    pdbids = []
    ligand_names = []
    total_loss, total_loss1, total_loss2 = 0, 0, 0
    probs_ls1, batch_ls1,  dist_ls1 = [], [], []
    probs_ls2, batch_ls2,  dist_ls2 = [], [], []
    batch_size_ls = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            data = batch_data.to(device)
            pi1, sigma1, mu1, dist1, batch1, pi2, sigma2, mu2, dist2, batch2 = model(data)
            if pred:
                batch_size = data['peptide'].batch.max()+1
                batch_size = batch_size.item()
                batch_size_ls.append(batch_size)

                prob1 = calculate_probablity_sv(pi1, sigma1, mu1, dist1)  # cb-cb
                probs_ls1.append(prob1)
                batch_ls1.append(batch1)
                dist_ls1.append(dist1)

                prob2 = calculate_probablity_sv(pi2, sigma2, mu2, dist2)  # min-min
                probs_ls2.append(prob2)
                batch_ls2.append(batch2)
                dist_ls2.append(dist2)

                pdbids.extend(data.pdb_id)
                try:
                    ligand_names.extend(data.ligand_name)
                except:
                    ligand_names.extend(data.pdb_id)

            else:
                gm_loss_true1 = gm_loss_fn_sv(pi1, sigma1, mu1, dist1)
                gm_loss_true1 = gm_loss_true1[torch.where(dist1 <= dist_threhold1)[0]].mean().float() 

                gm_loss_true2 = gm_loss_fn_sv(pi2, sigma2, mu2, dist2)
                gm_loss_true2 = gm_loss_true2[torch.where(dist2 <= dist_threhold2)[0]].mean().float() 

                batch_size = data['peptide'].batch.max()+1
                batch_size = batch_size.item()
                total_loss1 += gm_loss_true1.item()*batch_size
                total_loss2 += gm_loss_true2.item()*batch_size
                total_loss += (gm_loss_true1.item()*batch_size + gm_loss_true2.item()*batch_size)
                del data, pi1, sigma1, mu1, dist1, batch1, pi2, sigma2, mu2, dist2, 
                batch2,  gm_loss_true1, gm_loss_true2
                torch.cuda.empty_cache()
    if pred:
        return pdbids, ligand_names, probs_ls1, batch_ls1,  dist_ls1, probs_ls2, batch_ls2,  dist_ls2,  batch_size_ls 

    else:
        return total_loss/len(data_loader.dataset), total_loss1/len(data_loader.dataset), total_loss2/len(data_loader.dataset)
    

   

    
