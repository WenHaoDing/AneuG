import torch
import os
import pickle


class GHDDataset(torch.utils.data.Dataset):
    def __init__(self, ghd_chk_root, ghd_run, ghd_chk_name, ghd_reconstruct, cases, withscale=False, normalize=True):
        self.ghd_chk_root = ghd_chk_root
        self.ghd_run = ghd_run
        self.ghd_chk_name = ghd_chk_name
        self.ghd_reconstruct = ghd_reconstruct
        self.withscale = withscale
        self.cases = cases
        # get eigvec
        self.GHD_eigvec = self.ghd_reconstruct.canonical_ghd.GBH_eigvec
        # assemble dataset
        self.updated_cases, self.ghd, self.alignment, self.scale = self.assemble()
        # normalize
        self.normalize = normalize
        self.mean, self.std = self.normalize_()
        
    def assemble(self, print_jnl=False):
        updated_cases = []
        ghd = []
        alignment = []
        scale = []
        for case in self.cases:
            ghd_checkpoint = os.path.join(self.ghd_chk_root, case, self.ghd_run, self.ghd_chk_name)
            if os.path.exists(ghd_checkpoint):
                with open(ghd_checkpoint, 'rb') as f:
                    ghd_chk = pickle.load(f)
                ghd.append(ghd_chk['GHD_coefficient'].view(-1))
                R, s, T = ghd_chk['R'], ghd_chk['s'].abs(), ghd_chk['T']
                alignment.append(torch.cat((R.view(-1), s.view(-1), T.view(-1))).detach())
                scale.append(s.view(-1).detach())
                updated_cases.append(case)
            else:
                if print_jnl:
                    print("GHD checkpoint not found for case: ", case)
        print("{} cases out of {} have GHD checkpoint".format(len(updated_cases), len(self.cases)))
        return updated_cases, ghd, alignment, scale
    
    def normalize_(self):
        if self.withscale:
            dataset = torch.stack([torch.cat([ghd, scale]) for ghd, scale in zip(self.ghd, self.scale)], dim=0)
        else:
            dataset = torch.stack(self.ghd, dim=0)
        mean = dataset.mean(dim=0, keepdim=True)
        std = dataset.std(dim=0, keepdim=True) + 0.01
        return mean, std
    
    def __len__(self):
        return len(self.updated_cases)
    
    def __getitem__(self, idx):
        # case = self.updated_cases[idx]
        ghd = self.ghd[idx]
        scale = self.scale[idx]
        x = torch.cat([ghd, scale]) if self.withscale else ghd
        if self.normalize:
            x = (x - self.mean) / self.std
        return x.view(-1)
    
    def de_normalize(self, x):
        if not self.normalize:
            return x
        else:
            return x * self.std.to(x.device) + self.mean.to(x.device)
    
    def get_dim(self):
        x = self.__getitem__(0)
        if self.withscale:
            return x.shape[0] - 1
        else:
            return x.shape[0]
    
    def get_mean_std(self):
        if self.withscale:
            return self.mean[:, :-1], self.std[:, :-1]
        else:
            return self.mean, self.std
    
    def get_scale_mean_std(self):
        if self.withscale:
            return self.mean[:, -1], self.std[:, -1]
        else:
            return None, None
    
    def denorm_scale(self, scale):
        assert self.withscale
        return scale * self.std[:, -1].to(scale.device) + self.mean[:, -1].to(scale.device)
    

class CenterlineDataset(torch.utils.data.Dataset):
    def __init__(self, cl_chk_root: str, normalize=True, toss_threshold=0.005, device=torch.device('cuda:0')):
        self.cl_chk_root = cl_chk_root
        self.toss_threshold = toss_threshold
        self.cases, self.data, self.num_branch, self.num_fourier, self.fourier_per_branch = self.assemble()
        self.normalize = normalize
        self.norm_dict = self.normalize_()
        self.device = device

    def assemble(self, print_jnl=False):
        chk_files = [os.path.join(self.cl_chk_root, file) for file in os.listdir(self.cl_chk_root) if file.endswith('.pth')]
        cases = []
        start_end_vector = []
        split_centerline = []
        branch_length = []
        fouriers = []
        ghd = []
        scale = []
        relative_directions = []
        accurate_tangent = []

        toss_cases = []
        for chk_file in chk_files:
            chk = torch.load(chk_file)
            case = chk['label']
            if chk['fitting_loss']<self.toss_threshold:
                cases.append(chk['label'])
                # start_end_vector.append(chk['start_end_vector'])
                split_centerline.append(chk['split_centerline'])
                branch_length.append(torch.stack(chk['branch_length']).unsqueeze(0))  # [1, num_branch]
                fouriers.append(chk['fouriers'])
                ghd.append(chk['ghd'].unsqueeze(0))
                scale.append(chk['scale'].unsqueeze(0))
                relative_directions.append(chk['relative_directions'])
                accurate_tangent.append(torch.cat(chk['accurate_tangent']))
            else:
                toss_cases.append(chk['label'])
        temp = fouriers[0]
        num_branch, num_fourier, fourier_per_branch = temp.shape
        relative_directions = torch.cat([ten.view(1, 4*num_branch) for ten in relative_directions], dim=0)  # [1, 4*num_branch]
        ghd = torch.cat(ghd)
        fouriers = torch.cat([tensor_.view(1, num_branch*num_fourier*fourier_per_branch) for tensor_ in fouriers])
        branch_length = torch.cat(branch_length)
        print("{} cases have been loaded, {} cases have been tosses due to bad fitting".format(len(cases), len(toss_cases)))
        data = {'ghd': ghd, 'relative_directions': relative_directions, 'fouriers': fouriers, 'branch_length': branch_length,
                'accurate_tangent': accurate_tangent, 'scale': torch.cat(scale)}
        self.split_centerline = split_centerline
        return cases, data, num_branch, num_fourier, fourier_per_branch
    
    def normalize_(self):
        norm_dict = {}
        for key, value in self.data.items():
            if key != 'accurate_tangent':
                mean = value.mean(dim=0, keepdim=True)
                std = value.std(dim=0, keepdim=True) + 0.01
                norm_dict[key] = (mean, std)
        return norm_dict

    def de_normalize(self, data_dict):
        if not self.normalize:
            pass
        else:
            for key in [key_ for key_ in data_dict.keys() if key_ != 'accurate_tangent']:
                mean, std = self.norm_dict[key]
                data_dict[key] = data_dict[key] * std.to(data_dict[key].device) + mean.to(data_dict[key].device)
        return data_dict

    def __len__(self):
        return len(self.cases)
    
    def __getitem__(self, idx):
        data_dict = {}  
        for key in self.data.keys():
            if key != 'accurate_tangent':
                mean, std = self.norm_dict[key]
                data_dict[key] = self.data[key][idx].view(1, -1)
                data_dict[key] = ((self.data[key][idx] - mean) / std if self.normalize else self.data[key][idx]).squeeze(0)
            else:
                data_dict[key] = self.data[key][idx]
            data_dict[key] = data_dict[key].to(self.device)
        return data_dict
    
    def return_norm_dict(self, device):
        norm_dict = {}
        for key in self.norm_dict.keys():
            mean, std = self.norm_dict[key]
            norm_dict[key] = (mean.to(device), std.to(device))
        return norm_dict
        
    





        

        





        

        

