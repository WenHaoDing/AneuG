import pytorch3d.io
import pytorch3d
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
import os
import sys
import pickle
from ops.mesh_geometry import MeshThickness
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.ghb_utils import lazy_plot_meshes
from utils_oa.fn_opening_alignment import opening_alignment_pca
from registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
from utils_oa.fn_massive import opening_alignment_pca, Opening_Alignment_With_Differentiable_Centreline
from utils_oa.fn_surface_alignment import surface_alignment_pca
from fn_ghb.Graph_Harmonic_Fitting_plugin import Graph_Harmonic_Deform_opening_alignment_dynamic, Graph_Harmonic_Deform
from losses import Mesh_loss_opening_alignment, Mesh_loss_surface_partition_alignment, Mesh_loss_differentiable_occupancy
from losses import Mesh_loss_do_differentiable_centreline, Mesh_loss
from torch.utils.tensorboard import SummaryWriter
from .logger import log_dict_printer
from .logger import viz_fitting_static, viz_fitting_debug
from .weighter import base_loss_weighter
from .dropper import Do_Dropper
from new_version.ops.mesh_geometry import Winding_Occupancy
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes


def fit_ghd(args, loss_weighting, hard_normalize=True, keep_size=True, canonical_chk=None):
    # intialize registration
    canonical, target = initailize_registration(args, hard_normalize=hard_normalize, keep_size=keep_size)

    # create graph fitter and losser
    canonical_fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical)
    if not os.path.exists(canonical_chk):
        chk = {'GBH_eigval': getattr(canonical_fitter, "GBH_eigvec").detach().cpu(),
               'GBH_eigvec': getattr(canonical_fitter, "GBH_eigvec").detach().cpu()}
        with open(canonical_chk, 'wb') as f:
                pickle.dump(chk, f)
    else:
        with open(canonical_chk, 'rb') as f:
            chk = pickle.load(f)
        for key_ in chk.keys():
            setattr(canonical_fitter, key_, chk[key_].to(torch.device(args.device)))

    mesh_losser = Mesh_loss_do_differentiable_centreline(args, canonical, target)

    query_points, do_gt = mesh_losser.get_static_mask_and_gt(style=args.do_style)
    if args.do_loss_type == "dice_loss_attention":
        print('using attention dice loss, calculating attention weight map now')
        mesh_losser.get_weights_attention(query_points, min_w=1.0, max_w=args.attention_max_w, smooth=args.attention_smooth, inspect=False)
    query_points, do_gt = query_points.to(torch.device(args.device)), do_gt.to(torch.device(args.device))

    # thickness loss
    thinknesser = MeshThickness(r=0.2, num_bundle_filtered=100, innerp_threshold=0.6, num_sel=25)

    # training manager
    log_path = os.path.join(args.save_root, args.name_target, args.meta)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    optimizer = torch.optim.AdamW([canonical_fitter.deformation_param, canonical_fitter.s, canonical_fitter.T, canonical_fitter.R],
                                  lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2500, gamma=0.75)
    writer = SummaryWriter(log_path)
    loss_weighter = base_loss_weighter(args, glo_loss_weighting=loss_weighting, style=args.weighter_style)
    do_dropper = Do_Dropper(args, getattr(mesh_losser, "weights_attention"), drop_num=25, drop_rate=0.75)
    use_dropper = getattr(args, "use_do_dropper", 0)
    print("using do dropper") if use_dropper == 1 else print("using static do")
    query_points_update, do_gt_update = query_points, do_gt

    # main_loop
    for epoch in range(args.epochs):
        warped_mesh, warped_openings = canonical_fitter.forward_with_opening_alignment()
        loc_loss_weighting = loss_weighter.easy_weighting(epoch)  # update loss weighting
        if use_dropper == 1:
            do_index, update_do = do_dropper.forward(epoch)
        else:
            do_index, update_do = do_dropper.forward(epoch)
        if update_do:
            query_points_update, do_gt_update = query_points[do_index].clone(), do_gt[do_index].clone()  # update query points and do gt
        loss_dict = mesh_losser.forward_do_dcforward_opa_do(warped_mesh, warped_openings, loc_loss_weighting,
                                                            query_points_update, do_gt_update, do_index)
        # thickness loss
        if "loss_thickness" in loss_weighting:
            thickness_dict, thickness, _, sign = thinknesser.forward(warped_mesh)
            mask_thickness = torch.where(thickness.abs() > 0.1, torch.zeros_like(thickness), torch.ones_like(thickness))
            signed = torch.sign(sign)
            loss_thickness = (F.relu(0.04 - thickness * signed) + F.relu(0.01 - thickness_dict * signed))*mask_thickness
            loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask_thickness).mean()
            loss_dict["loss_thickness"] = loss_thickness

        total_loss = torch.zeros(1, device=torch.device(args.device))
        log_dict = {}
        log_dict['epoch'] = epoch
        for term, loss in loss_dict.items():
            if term not in ['loss_openings_p', 'loss_openings_n']:
                total_loss += loss * loc_loss_weighting[term]
                writer.add_scalar('Train/' + term, (loss * loc_loss_weighting[term]).cpu().item(), epoch)
                log_dict[term] = (loss * loc_loss_weighting[term]).cpu().item()
            else:
                total_loss += torch.sum(torch.stack(loss), dim=0) * loc_loss_weighting[term]
                writer.add_scalar('Train/' + term,
                                  (torch.sum(torch.stack(loss), dim=0) * loc_loss_weighting[term]).cpu().item(),
                                  epoch)
                log_dict[term] = (torch.sum(torch.stack(loss), dim=0) * loc_loss_weighting[term]).cpu().item()
        if epoch % args.log_freq == 0:
            log_dict_printer(log_dict)
        if epoch % (4 * args.log_freq) == 0:
            print(args.name_target)

        # logging
        viz_fitting_static(epoch, log_path, warped_mesh, getattr(mesh_losser, "target_mesh"), args)

        # gradient descent
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # saving chk
        if epoch % args.chk_freq == 0 and epoch != 0:
            chk_path = os.path.join(log_path, "ghb_fitting_checkpoint_" + str(round(epoch / args.chk_freq)) + ".pkl")
            chk = {'R': getattr(canonical_fitter, "R").detach().cpu(),
                   's': getattr(canonical_fitter, 's').detach().cpu().abs(),
                   'T': getattr(canonical_fitter, 'T').detach().cpu(),
                   'GHD_coefficient': getattr(canonical_fitter, 'deformation_param').detach().cpu()}
            with open(chk_path, 'wb') as f:
                pickle.dump(chk, f)
            print('GHB fitting results have been saved to {}'.format(chk_path))

    # saving
    chk_path = os.path.join(log_path, "ghb_fitting_checkpoint.pkl")
    chk = {'R': getattr(canonical_fitter, "R").detach().cpu(),
           's': getattr(canonical_fitter, 's').detach().cpu().abs(),
           'T': getattr(canonical_fitter, 'T').detach().cpu(),
           'GHD_coefficient': getattr(canonical_fitter, 'deformation_param').detach().cpu()}
    with open(chk_path, 'wb') as f:
        pickle.dump(chk, f)
    print('GHB fitting results have been saved to {}'.format(chk_path))

def initailize_registration(args, hard_normalize=True, keep_size=True):
    print("Bold opening normal sorting = {}".format(True if args.op_bold == 1 else False))
    canonical = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_template, args.name_canonical)
    canonical.load_checkpoint_opa(None)
    canonical.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    canonical.load_checkpoint_centreline(None, redo=False)
    norm_canonical = torch.max(torch.norm(getattr(canonical, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item() * 1.10 if hard_normalize else 10.0
    if keep_size:
        norm_canonical = 2.50 * norm_canonical
        print("keeping same size ratio, which means canonical is normalized using 2.50 * radius")
    canonical.class_normalize(norm=norm_canonical)
    canonical.centreline_clean(radius=0.5 / norm_canonical)

    target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
    target.load_checkpoint_opa(None)
    target.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    target.load_checkpoint_centreline(None, redo=False)
    norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(),
                                       dim=-1)).detach().item() * 1.10 if hard_normalize else 7.5
    norm_target = norm_canonical if keep_size else norm_target
    target.class_normalize(norm=norm_target)
    target.centreline_clean(radius=0.5 / norm_target)
    print("canonical and target Meshes have been normalized using radius={} and {}".format(norm_canonical, norm_target))
    return canonical, target

def Mesh_normalize(mesh: Meshes, extra_factor=0.1):
    norm = torch.max(torch.norm(mesh.verts_packed(), dim=-1)).detach().item() * (1+extra_factor)
    original_mesh_verts = mesh.verts_padded().float()
    updated_mesh_verts = original_mesh_verts / norm
    normalized_mesh = mesh.update_padded(updated_mesh_verts)
    return normalized_mesh
