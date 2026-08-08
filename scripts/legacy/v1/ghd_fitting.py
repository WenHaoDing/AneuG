import argparse
from ghd.fitting.fitter import fit_ghd
import os
import logging
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
import torch
import random

# conf
epochs = 15000
chk_num = 4  # number of checkpoints during fitting
register = True  
parser = argparse.ArgumentParser("ghb_fitting_oa")
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--root_template', type=str, default='./checkpoints/alignment_male_medium')
parser.add_argument('--root_target', type=str, default='./checkpoints/alignment_male_medium')
parser.add_argument('--name_canonical', type=str, default='canonical_typeB')
parser.add_argument('--name_target', type=str, default='AN213_full_clean')
parser.add_argument('--viz_freq', type=int, default=200)
parser.add_argument('--chk_freq', type=int, default=round(epochs / chk_num))
parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.00075)
parser.add_argument('--num_op', type=int, default=3)
parser.add_argument('--num_Basis', type=int, default=13 ** 2)
parser.add_argument('--mix_lap_weights', type=list, default=[1.0, 0.1, 0.1])
parser.add_argument('--sample_num', type=int, default=int(2.5e5))
parser.add_argument('--op_sample_num', type=int, default=int(1e3))
parser.add_argument('--op_clean_threshold', type=float, default=0.2)
parser.add_argument('--op_bold', type=int, default=0)  # confidence that trimesh offers uniform mesh normal directions
parser.add_argument('--save_root', type=str, default='./checkpoints/ghb_fitting_male_medium')
parser.add_argument('--meta', type=str, default='cut_446_decreasing_centrelineloss_10')
parser.add_argument('--epochs', type=int, default=int(epochs))
parser.add_argument('--num_sp', type=int, default=2)
parser.add_argument('--do_dpi', type=int, default=4)
parser.add_argument('--do_style', type=str, default='number_control_v2')
parser.add_argument('--do_loss_type', type=str, default='dice_loss_attention')
parser.add_argument('--use_do_dropper', type=int, default=0)
parser.add_argument('--attention_max_w', type=float, default=3.0)
parser.add_argument('--attention_smooth', type=float, default=0.02)
parser.add_argument('--do_number', type=int, default=25000)
parser.add_argument('--weighter_style', type=str, default='strategy_v1_linear')
args = parser.parse_args()

loss_weighting = {
    'loss_do': 1.0,
    'loss_p0': 1.0 * 1, 'loss_n1': 0.8 * 1,
    'loss_laplacian': 0.1, 'loss_edge': 0.1, 'loss_consistency': 0.1,
    'loss_rigid': 100.0,
    'loss_openings_p': 5,
    'loss_openings_n': 0.1,
    'loss_diff_centreline': 10.0
}

label_list = [label for label in os.listdir(args.root_target) if
              os.path.isdir(os.path.join(args.root_target, label)) and label != args.name_canonical]
exclude_list = []
random.shuffle(label_list)

# perform registration for opa classes and differentiable centrelines
if register:
    for label in label_list:
        args.name_target = label
        if (os.path.exists(os.path.join(args.root_target, args.name_target, "opa_checkpoint.pkl")) and os.path.exists(os.path.join(args.root_target, args.name_target, "diff_centreline_checkpoint.pkl"))):
            print("Registration for case {} has been found, skipping".format(label))
        else:
            print("Registration for case {} not found".format(label))
            target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
            target.load_checkpoint_opa(None, redo=False)
            target.load_checkpoint_centreline(None, redo=False)
            norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item()
            target.class_normalize(norm=norm_target)
            target.centreline_clean(radius=0.5 / norm_target)
            target.visualize_centreline(norm_target)

# perform ghd fitting
for label in label_list:
    args.name_target = label
    print('Now performing ghd fitting for case {}'.format(label))
    if not os.path.exists(os.path.join(args.save_root, args.name_target, args.meta, "ghd_fitting_checkpoint.pkl")):
        copied_loss_weighting = loss_weighting.copy()
        fit_ghd(args, copied_loss_weighting, hard_normalize=True, keep_size=True)
    else:
        print("Skipping ghd fitting for case {}".format(label))