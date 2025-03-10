import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
from models.mesh_plugins import MeshPlugins
from models.ghd_reconstruct import GHD_Reconstruct


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, withscale=False):
        super(VAE, self).__init__()
        self.withscale = withscale
        self.input_dim = input_dim if not withscale else input_dim+1
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, self.input_dim)

    def encode(self, x, scale=None):
        if scale is not None:
            x = torch.cat((x, scale), dim=1)
        x = self.fc1(x)
        x = self.res1(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, strip_scale=False):
        x = self.fc3(z)
        x = self.res2(x)
        x = self.fc4(x)
        if self.withscale:
            if not strip_scale:
                return x[:, :-1], x[:, -1:]
            else:
                return x[:, :-1]
        else:
            return x

    def forward(self, x, scale=None):
        mu, logvar = self.encode(x, scale)
        z = self.reparameterize(mu, logvar)
        if self.withscale:
            return *self.decode(z), mu, logvar
        else:
            return self.decode(z), mu, logvar


class ConditionalGHDVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, cond_dim=2):
        super(ConditionalGHDVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim+cond_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = self.fc1(x)
        x = self.res1(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, cond):
        z = torch.cat((z, cond), dim=1)
        x = self.fc3(z)
        x = self.res2(x)
        x = self.fc4(x)
        return x

    def forward(self, x, cond):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, cond), mu, logvar

    def forward_encoder(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class ConditionalVAE4Fouriers(nn.Module):
    def __init__(self, num_branch=3, num_fourier=8, fourier_per_branch=2, num_basis=4**2, 
                 hidden_dim=64, latent_dim=16, dropout=0, 
                 tangent_encoding=True, ghd_reconstruct: GHD_Reconstruct=None, mesh_plugin: MeshPlugins=None,
                 cpcd_reconstruct=None,
                 norm_dict: dict=None, normalize=True,
                 ghd_encoding=False,
                 withscale=False,
                 l_condition=False):
        super(ConditionalVAE4Fouriers, self).__init__()

        self.withscale = withscale
        # branch length + reltaive direcations + fouriers
        self.num_branch = num_branch
        self.num_fourier = num_fourier
        self.fourier_per_branch = fourier_per_branch
        self.num_basis = num_basis

        # ghd encoding net (when using all ghd information, we compress to 3 features)
        self.ghd_encoding_net = nn.Sequential(nn.Linear(num_basis*3, 3), nn.ReLU(), nn.Linear(3, 3))

        self.input_dim = num_branch+4*num_branch+num_branch*num_fourier*fourier_per_branch
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.tangent_encoding = tangent_encoding

        self.ghd_reconstruct = ghd_reconstruct
        self.mesh_plugin = mesh_plugin  # for tangent encoding

        self.cpcd_reconstruct = cpcd_reconstruct

        self.norm_dict = norm_dict      # for de-normalize data dict input
        self.normalize = normalize
        
        # when using ghd encoding, we compress ghd to 3 features
        self.ghd_encoding = ghd_encoding
        if self.ghd_encoding:
            self.condition_dim = 3
        else:
            self.condition_dim = num_basis*3
        # if tangent encoding, we add 3 vectors as additional features (which are functions of ghd)
        if tangent_encoding:
            self.condition_dim += num_branch*3
        if self.withscale:
            self.condition_dim += 1

        # Encoder layers
        self.fc1 = nn.Linear(self.input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.fc21_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout21_1 = nn.Dropout(dropout)
        self.fc21_2 = nn.Linear(hidden_dim, latent_dim)

        self.fc22_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout22_1 = nn.Dropout(dropout)
        self.fc22_2 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fcd1 = nn.Linear(latent_dim + self.condition_dim, hidden_dim)
        self.resd1 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d1 = nn.Dropout(dropout)
        self.fcd2 = nn.Linear(hidden_dim, hidden_dim)
        self.resd2 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d2 = nn.Dropout(dropout)
        self.fcd3 = nn.Linear(hidden_dim, hidden_dim)
        self.resd3 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d3 = nn.Dropout(dropout)
        self.fce = nn.Linear(hidden_dim, self.input_dim)
        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def encode(self, x):
        x = self.fc1(x)
        x = self.res1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.res2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.res3(x)
        x = self.dropout3(x)
        return self.dropout21_1(self.fc21_2(self.fc21_1(x))), self.dropout22_1(self.fc22_2(self.fc22_1(x)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, condition):
        if self.ghd_encoding:
            if self.tangent_encoding:
                shift = 3*self.num_basis+1 if self.withscale else 3*self.num_basis
                tangent = condition[:, :-shift]
                ghd_condition = condition[:, -shift:]
                if self.withscale:
                    ghd_condition, scale = ghd_condition[:, :-1], ghd_condition[:, -1:]
                    ghd_condition = self.ghd_encoding_net(ghd_condition)
                    condition = torch.cat((tangent, ghd_condition, scale), dim=-1)
                else:
                    ghd_condition = ghd_condition
                    ghd_condition = self.ghd_encoding_net(ghd_condition)
                    condition = torch.cat((tangent, ghd_condition), dim=-1)
            else:
                if self.withscale:
                    ghd_condition, scale = condition[:, :-1], condition[:, -1:]
                    ghd_condition = self.ghd_encoding_net(ghd_condition)
                    condition = torch.cat((ghd_condition, scale), dim=-1)
                else:
                    condition = self.ghd_encoding_net(condition)
        x = self.fcd1(torch.cat((z, condition), dim=1))
        x = self.resd1(x)
        x = self.dropout_d1(x)
        x = self.fcd2(x)
        x = self.resd2(x)
        x = self.dropout_d2(x)
        x = self.fcd3(x)
        x = self.resd3(x)
        x = self.dropout_d3(x)
        x = self.fce(x)
        return x

    def forward(self, data_dict: dict):
        x, cond, tangent_diff, start_points, accurate_tangent = self.translate_input(data_dict, cheat_tangent=True)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        # reconstruct cpcd
        true_src = self.translate_output(x)
        pred_src = self.translate_output(recon)
        _, cpcd_glo_true, cpcd_tangent_glo_true = self.cpcd_reconstruct.forward_cpcd(accurate_tangent, start_points, *true_src)  # [B, num_branch, dpi, 3]
        _, cpcd_glo_pred, cpcd_tangent_glo_pred = self.cpcd_reconstruct.forward_cpcd(accurate_tangent, start_points, *pred_src)  # [B, num_branch, dpi, 3]
        return x, recon, mu, logvar, cpcd_glo_true, cpcd_glo_pred, cpcd_tangent_glo_true, cpcd_tangent_glo_pred, accurate_tangent
    
    def forward_encoder(self, data_dict: dict):
        x, cond, tangent_diff, start_points, accurate_tangent = self.translate_input(data_dict, cheat_tangent=True)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
    
    def translate_input(self, data_dict: dict, cheat_tangent=False):
        """
        data_dict = {'ghd': ghd, 'relative_directions': relative_directions, 'fouriers': fouriers, 'branch_length': branch_length, scale: scale}
        set cheat_tangent to True for data_dict in the dataset
        Note: Due to ghd morphoing trying to capture the details, tangent vectors estimated using the wave method will be inaccurate. However, since we only need the 
        tangent vectors for condition, they don't have to be differentiable. Therefore, we can "cheat" by using the accurate tangent vectors estimated using the difference method.
        Only regulization loss require the tangent vectors to be differentiable.
        """
        ghd, relative_directions, fouriers, branch_length = data_dict['ghd'], data_dict['relative_directions'], data_dict['fouriers'], data_dict['branch_length']
        
        x = torch.cat((branch_length, relative_directions, fouriers), dim=1)
        if self.withscale:
            scale = data_dict['scale']
        else:
            scale = None
        cond, tangent_diff, start_points = self.mask_ghd_as_cond(ghd, scale=scale)
        if cheat_tangent:
            accurate_tangent = data_dict['accurate_tangent']
            return x, cond, tangent_diff, start_points, accurate_tangent
        else:
            x, cond, tangent_diff, start_points
    
    def mask_ghd_as_cond(self, ghd, cheat_tangent=False, accurate_tangent=None, scale=None):
        B = ghd.shape[0]
        masked_ghd = ghd.reshape(B, -1, 3)
        masked_ghd = masked_ghd[:, :self.num_basis, :].reshape(B, -1)

        if not self.tangent_encoding:
            cond = masked_ghd
        else:
            if self.normalize:
                ghd = ghd * self.norm_dict['ghd'][1] + self.norm_dict['ghd'][0]
                if self.withscale:
                    scale = scale * self.norm_dict['scale'][1] + self.norm_dict['scale'][0]
            Meshes_ = self.ghd_reconstruct.ghd_forward_as_Meshes(ghd, denormalize_shape=False, scale=scale)
            _, _, tangent_diff, start_points = self.mesh_plugin.mesh_forward_tangents(Meshes_, return_start_points=True)
            if cheat_tangent:
                cond = torch.cat((accurate_tangent.view(B, self.num_branch*3), masked_ghd), dim=1)
            else:
                cond = torch.cat((tangent_diff.view(B, self.num_branch*3), masked_ghd), dim=1)
        if self.withscale:
            assert scale is not None, "Scale is required for withscale"
            cond = torch.cat((cond, scale), dim=1)
        return cond, tangent_diff, start_points

    def translate_output(self, y):
        """
        from vae input-output vector to cpcd features
        we use this for both true and pred cpcd reconstruction
        """
        # extract
        B = y.shape[0]
        branch_length = y[:, :self.num_branch]
        relative_directions = y[:, self.num_branch:self.num_branch+4*self.num_branch]
        fouriers = y[:, self.num_branch+4*self.num_branch:]
        # de-normalize
        if self.normalize:
            branch_length = branch_length * self.norm_dict['branch_length'][1] + self.norm_dict['branch_length'][0]
            relative_directions = relative_directions * self.norm_dict['relative_directions'][1] + self.norm_dict['relative_directions'][0]
            fouriers = fouriers * self.norm_dict['fouriers'][1] + self.norm_dict['fouriers'][0]

        # reshape
        branch_length = branch_length.view(B, self.num_branch)
        relative_directions = relative_directions.view(B, self.num_branch, 4)
        fouriers = fouriers.view(B, self.num_branch, self.num_fourier, self.fourier_per_branch)
        return branch_length, relative_directions, fouriers
    
    def generate(self, ghd, scale=None, z=None, tangent_shift=None):
        """
        we take actual ghd and actual scale as input
        cpcd_glo: [B, num_branch, dpi, 3]
        cpcd_tangent_glo: [B, num_branch, 3]
        """
        B = ghd.shape[0]
        if z is None:
            z = 1.0 * torch.randn(B, self.latent_dim).to(ghd.device)
        else:
            z = z.to(ghd.device)
        if self.normalize:
            ghd = (ghd - self.norm_dict['ghd'][0]) / self.norm_dict['ghd'][1]
        if self.withscale:
            assert scale is not None, "Scale is required for withscale"
            if self.normalize:
                scale = (scale - self.norm_dict['scale'][0]) / self.norm_dict['scale'][1]
        else:
            scale = None
        cond, tangent_diff, start_points = self.mask_ghd_as_cond(ghd, cheat_tangent=False, scale=scale)
        recon = self.decode(z, cond)
        branch_length, relative_directions, fouriers = self.translate_output(recon)
        cpcd_loc, cpcd_glo, cpcd_tangent_glo = self.cpcd_reconstruct.forward_cpcd(tangent_diff, start_points, branch_length, relative_directions, fouriers, tangent_shift)
        return cpcd_glo, cpcd_tangent_glo, tangent_diff
    
    def generate_controlled(self, ghd, scale=None, z=None, tangent_shift=None):
        """
        we take actual ghd and actual scale as input
        cpcd_glo: [B, num_branch, dpi, 3]
        cpcd_tangent_glo: [B, num_branch, 3]
        """
        B = ghd.shape[0]
        if z is None:
            z = 1.0 * torch.randn(B, self.latent_dim).to(ghd.device)
        else:
            z = z.to(ghd.device)
        if self.normalize:
            ghd = (ghd - self.norm_dict['ghd'][0]) / self.norm_dict['ghd'][1]
        if self.withscale:
            assert scale is not None, "Scale is required for withscale"
            if self.normalize:
                scale = (scale - self.norm_dict['scale'][0]) / self.norm_dict['scale'][1]
        else:
            scale = None
        cond, tangent_diff, start_points = self.mask_ghd_as_cond(ghd, cheat_tangent=False, scale=scale)
        cond = cond.mean(dim=0, keepdim=True).repeat(B, 1)
        recon = self.decode(z, cond)
        branch_length, relative_directions, fouriers = self.translate_output(recon)
        cpcd_loc, cpcd_glo, cpcd_tangent_glo = self.cpcd_reconstruct.forward_cpcd(tangent_diff, start_points, branch_length, relative_directions, fouriers, tangent_shift)
        return cpcd_glo, cpcd_tangent_glo, tangent_diff



class CPCDReconstruct(object):
    """
    Centerline Point Cloud Reconstructer
    """
    def __init__(self, num_branch=3, num_fourier=8, fourier_per_branch=2, device=torch.device('cuda:0'), dpi=250):
        self.num_branch = num_branch
        self.num_fourier = num_fourier
        self.fourier_per_branch = fourier_per_branch
        self.device = device
        self.dpi = dpi
        self.shape_modes_x, self.shape_modes_yz, self.tangent_shape_modes_yz = self.get_loc_shape_modes()

    def get_loc_shape_modes(self):
        # shape modes [num_fourier, dpi]
        wl_ratio = torch.Tensor([0.5 * i for i in range(1, self.num_fourier + 1)]).unsqueeze(-1).to(self.device)  # [num_fourier, 1]
        shape_modes_x = torch.linspace(0, 1, self.dpi).unsqueeze(0).to(self.device)  # [1, dpi]
        # we repeat twice since we have y and z
        shape_modes_yz = torch.sin(2*np.pi*wl_ratio*shape_modes_x).unsqueeze(-2).repeat(1, self.fourier_per_branch, 1).to(self.device)  # [num_fourier, fourier_per_branch, dpi]
        # transformation of middle system to loc will happen in forward_cpcd
        tangent_shape_modes_yz = (2*np.pi*wl_ratio.unsqueeze(-1) * (torch.cos(2*np.pi*wl_ratio*shape_modes_x).unsqueeze(-2).repeat(1, self.fourier_per_branch, 1))).to(self.device)  # [num_fourier, fourier_per_branch, dpi]
        # we also calculate tangent shape modes [num_fourier, 1]
        # tangent_shape_modes_yz_ = 2*np.pi*wl_ratio.to(self.device)  # [num_fourier, 1]
        return shape_modes_x, shape_modes_yz, tangent_shape_modes_yz

    def forward_cpcd(self, tangent, start_points, branch_length, relative_directions, fouriers, tangent_shift=[0, 0.06, 0]):
        """
        reconstruct world coordinate system centerline points with fourier
        fouriers -> cpcd in world coordinate system
        tangent: [B, num_branch, 3]  (could be tangent_diff or tangent_pca)
        branch_length: [B, num_branch]
        relative_directions: [B, num_branch, 4]
        fouriers: [B, num_branch, num_fourier, fourier_per_branch]
        start_points: [B, num_branch, 3]
        output
        cpcd_glo: [B, num_branch, dpi, 3]
        cpcd_tangent_glo: [B, num_branch, dpi, 3]
        we fix reconstruction dpi=250
        """
        # get cross in wcs
        rds_vector = relative_directions[:, :, :-1]
        rds_module = relative_directions[:, :, -1:]
        rds_mapped = rds_vector * rds_module  # [B, num_branch, 3]
        # directions in wcs = loc system x vector = rds_mapped (strength controled by module) + tangent
        loc_sys_x = rds_mapped + tangent
        loc_sys_x = loc_sys_x / torch.norm(loc_sys_x, dim=2, keepdim=True)  # [B, num_branch, 3]
        # -> average cross product [B, 1, 3]
        average_cross = torch.mean(torch.cross(tangent, torch.cat((tangent[:, 1:, :], tangent[:, :1, :]), dim=1), dim=-1), dim=1, keepdim=True)
        average_cross = average_cross / torch.norm(average_cross, dim=2, keepdim=True)
        # get world_to_loc_matrix
        # [B, 1, 3] - [B, num_branch, 3] * {[B, 1, 3] @ [B, 3, num_branch] -> [B, num_branch, 1])} --> [B, num_branch, 3]
        loc_sys_z = average_cross - loc_sys_x * torch.einsum('blc,bcn->bln', average_cross, loc_sys_x.transpose(1, 2)).transpose(1, 2)
        loc_sys_z = loc_sys_z / torch.norm(loc_sys_z, dim=2, keepdim=True)
        loc_sys_y = torch.cross(loc_sys_z, loc_sys_x, dim=-1)
        loc_sys_y = loc_sys_y / torch.norm(loc_sys_y, dim=2, keepdim=True)
        l2w_matrix = torch.stack((loc_sys_x, loc_sys_y, loc_sys_z), dim=2)  # [B, num_branch, 3, 3]
        # p_local = p_world @ w2l_matrix
        # p_world = p_local @ l2w_matrix

        # [num_fourier, fourier_per_branch, dpi]
        # fouriers * shape modes [B, num_branch, num_fourier, fourier_per_branch] [num_fourier, fourier_per_branch, dpi] -> [B, num_branch, num_fourier, num_fourier, fourier_per_branch, dpi]
        cpcd_loc_yz = fouriers.unsqueeze(-1) * self.shape_modes_yz.unsqueeze(0).unsqueeze(0)  # [B, num_branch, num_fourier, fourier_per_branch, dpi]
        cpcd_loc_yz = cpcd_loc_yz.sum(dim=2)  # [B, num_branch, fourier_per_branch, dpi]
        # branch length * shape modes [B, num_branch] * [1, dpi] -> [B, num_branch, dpi]
        cpcd_loc_x = branch_length.unsqueeze(-1) * self.shape_modes_x.unsqueeze(0)
        cpcd_loc_x = cpcd_loc_x.unsqueeze(2)
        cpcd_loc = torch.cat((cpcd_loc_x, cpcd_loc_yz), dim=2).permute(0, 1, 3, 2)  # [B, num_branch, dpi, 3]

        # calculate tangenet vector 
        # branch_length: [B, num_branch]
        # fouriers * tangent shape modes [B, num_branch, num_fourier, fourier_per_branch] [num_fourier, 1] -> 
        # note: due to loc -> intermediate coordinate system transformation, we need to shrink by branch_length
        cpcd_tangent_loc_yz = fouriers.unsqueeze(-1) * self.tangent_shape_modes_yz.unsqueeze(0).unsqueeze(0)
        cpcd_tangent_loc_yz = cpcd_tangent_loc_yz.sum(dim=2)  # [B, num_branch, fourier_per_branch, dpi]
        cpcd_tangent_loc_yz = cpcd_tangent_loc_yz / branch_length.unsqueeze(-1).unsqueeze(-1)  # [B, num_branch, fourier_per_branch, dpi]
        B = cpcd_tangent_loc_yz.shape[0]
        cpcd_tangent_loc_x = torch.ones(B, self.num_branch, 1, self.dpi).to(self.device)
        cpcd_tangent_loc = torch.cat((cpcd_tangent_loc_x, cpcd_tangent_loc_yz), dim=2).permute(0, 1, 3, 2)  # [B, num_branch, dpi, fourier_per_branch+1]
        cpcd_tangent_loc = cpcd_tangent_loc / torch.norm(cpcd_tangent_loc, dim=-1, keepdim=True)
    
        # get world coordinate system cpcd
        # cpcd_loc * l2w_matrix [B, num_branch, dpi, 3] @ [B, num_branch, 3, 3]
        cpcd_glo = torch.einsum('bndc, bncl->bndl', cpcd_loc, l2w_matrix)  # [B, num_branch, dpi, 3]
        # add shift (just for visualization)
        cpcd_glo += start_points.unsqueeze(-2)
        # convert tangent
        cpcd_tangent_glo = torch.einsum('bndc, bncl->bndl', cpcd_tangent_loc, l2w_matrix)  # [B, num_branch, dpi, 3]
        # add tangent shift
        if tangent_shift is not None:
            cpcd_tangent_glo_initial = cpcd_tangent_glo[..., :1, :]
            tangent_shift_ = cpcd_tangent_glo_initial * torch.Tensor(tangent_shift).to(self.device).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
            cpcd_glo += tangent_shift_
        return cpcd_loc, cpcd_glo, cpcd_tangent_glo



def KL_divergence(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD


def save_checkpoint_gan(model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_G_state_dict': model_G.state_dict(),
        'model_D_state_dict': model_D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_state_dict': scheduler_D.state_dict()
    }
    torch.save(checkpoint, file_path)
    print(f'Checkpoint saved to {file_path}')


def load_checkpoint_gan(model_G, model_D, optimizer_G, optimizer_D, scheduler_G, scheduler_D, file_path, model_only=False, retain_epoch=False):
    checkpoint = torch.load(file_path)
    model_G.load_state_dict(checkpoint['model_G_state_dict'])
    model_D.load_state_dict(checkpoint['model_D_state_dict'])
    if not model_only:
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
    if not retain_epoch:
        epoch = 0
    else:
        epoch = checkpoint['epoch']
    print(f'Checkpoint loaded from {file_path} (epoch {epoch})')
    return epoch


def compute_gradient_penalty_dim2(D, real_samples, fake_samples, condition):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1))).cuda()
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, condition)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0).cuda(), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Discriminator_ConditionDot(nn.Module):
    def __init__(self, input_dim, args, use_dot=True):
        super(Discriminator_ConditionDot, self).__init__()
        if use_dot:
            self.fc1 = nn.Linear(input_dim+args.num_branch, 256)
        else:
            self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.LayerNorm(256)
        self.bn2 = nn.LayerNorm(128)
        self.relu = nn.LeakyReLU()
        self.use_dot = use_dot

    def forward(self, x_d, condition_d):
        B = x_d.shape[0]
        if self.use_dot:
            condition_d = condition_d.view((B, 3, 3))  # [B, 3*3] -> [B, 3, 3]
            start_to_end_d = x_d[:, :9].view((B, 3, 3))
            dot_product_matrix = torch.bmm(condition_d, start_to_end_d.transpose(1, 2))  # [B, 3, 3]
            dot_product = dot_product_matrix.diagonal(dim1=-2, dim2=-1)  # [B, 3]
            x_d = F.leaky_relu(self.bn1(self.fc1(torch.cat((x_d, condition_d.view((B, -1)), dot_product), dim=1))))
            x_d = F.leaky_relu(self.bn2(self.dropout(self.fc2(x_d))))
            x_d = self.fc3(x_d)
        else:
            x_d = F.leaky_relu(self.bn1(self.fc1(torch.cat((x_d, condition_d.view((B, -1))), dim=1))))
            x_d = F.leaky_relu(self.bn2(self.dropout(self.fc2(x_d))))
            x_d = self.fc3(x_d)
        return x_d  # validity
    

class VAE2(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE2, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.res3 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(p=0.1)
        self.fc21_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout21_1 = nn.Dropout(p=0.1)
        self.fc21_2 = nn.Linear(hidden_dim, latent_dim)
        self.fc22_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout22_1 = nn.Dropout(p=0.1)
        self.fc22_2 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fcd1 = nn.Linear(latent_dim, hidden_dim)
        self.resd1 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d1 = nn.Dropout(p=0.1)
        self.fcd2 = nn.Linear(hidden_dim, hidden_dim)
        self.resd2 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d2 = nn.Dropout(p=0.1)
        self.fcd3 = nn.Linear(hidden_dim, hidden_dim)
        self.resd3 = ResidualBlock(hidden_dim, hidden_dim)
        self.dropout_d3 = nn.Dropout(p=0.1)
        self.fce = nn.Linear(hidden_dim, input_dim)
        # self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def encode(self, x):
        x = self.fc1(x)
        x = self.res1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.res2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.res3(x)
        x = self.dropout3(x)
        return self.dropout21_1(self.fc21_2(self.fc21_1(x))), self.dropout22_1(self.fc22_2(self.fc22_1(x)))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fcd1(z)
        x = self.resd1(x)
        x = self.dropout_d1(x)
        x = self.fcd2(x)
        x = self.resd2(x)
        x = self.dropout_d2(x)
        x = self.fcd3(x)
        x = self.resd3(x)
        x = self.dropout_d3(x)
        x = self.fce(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class TransformerVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.res1 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.res2 = ResidualBlock(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = self.fc1(x)
        x = self.res1(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc3(z)
        x = self.res2(x)
        x = self.fc4(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
