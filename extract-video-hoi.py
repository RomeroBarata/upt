import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from utils import DataFactory
from upt import build_detector


KEEP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90, 91
]


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet101', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position-embedding', default='sine', type=str, choices=('sine', 'learned'))

    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument('--eos-coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--data-root', default='./hicodet')
    parser.add_argument('--human-idx', type=int, default=0)

    parser.add_argument('--device', default='cpu')
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument('--box-score-thresh', default=0.0, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=101, type=int)
    parser.add_argument('--nms', default=1.0, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--index', default=0, type=int)
    parser.add_argument('--action', default=None, type=int,
                        help="Index of the action class to visualise.")
    parser.add_argument('--action-score-thresh', default=0.2, type=float,
                        help="Threshold on action classes.")
    parser.add_argument('--image-path', default=None, type=str,
                        help="Path to an image file.")

    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video frames.')
    parser.add_argument('--object_detections_dir', type=str, required=True,
                        help='Directory containing all detections. E.g. '
                             '/home/romero/data/MPIICooking2/object-detection/detr')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Where to save detected HOIs. E.g. '
                             '/home/romero/data/MPIICooking2/human-object-interaction.')

    return parser


def load_tensor(filepath, device):
    tensor = torch.from_numpy(np.load(filepath))
    tensor = torch.unsqueeze(torch.unsqueeze(tensor, dim=0), dim=0).to(device)
    return tensor


def rescale_bb_from_original_to_zero_one(bbs, w, h):
    im_size = torch.reshape(torch.tensor([w, h, w, h], dtype=torch.float, device=bbs.device), [1, 1, 1, 4])
    bbs = bbs / im_size
    return bbs


def xyxy_to_cxcywh(bbs):
    cx = (bbs[:, :, :, 0:1] + bbs[:, :, :, 2:3]) / 2
    cy = (bbs[:, :, :, 1:2] + bbs[:, :, :, 3:4]) / 2
    w = bbs[:, :, :, 2:3] - bbs[:, :, :, 0:1]
    h = bbs[:, :, :, 3:4] - bbs[:, :, :, 1:2]
    bbs = torch.cat([cx, cy, w, h], dim=-1)
    return bbs


# TODO: Test this script on a video and visualise results. Everything working fine, run on full dataset.
def main():
    torch.set_num_threads(1)  # Increase this value to ~5 if using cpu
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    device = args.device
    # Load UPT model
    dataset = DataFactory(name=args.dataset, partition=args.partition, data_root=args.data_root)
    conversion = dataset.dataset.object_to_verb if args.dataset == 'hicodet' \
        else list(dataset.dataset.object_to_action.values())
    args.num_classes = 117 if args.dataset == 'hicodet' else 24
    upt = build_detector(args, conversion)
    upt.to(device)
    upt.eval()
    # Set up base dirs for provided info
    video_id = os.path.basename(args.frames_dir)
    bbs_dir = os.path.join(args.object_detections_dir, 'bounding_boxes', video_id)
    cls_logits_dir = os.path.join(args.object_detections_dir, 'classes_logits', video_id)
    features_dir = os.path.join(args.object_detections_dir, 'features', video_id)
    # Set up save dirs for extracted HOIs
    logits_save_dir = os.path.join(args.save_dir, 'classes_logits', video_id)
    os.makedirs(logits_save_dir, exist_ok=True)
    human_indices_save_dir = os.path.join(args.save_dir, 'human_indices', video_id)
    os.makedirs(human_indices_save_dir, exist_ok=True)
    obj_indices_save_dir = os.path.join(args.save_dir, 'object_indices', video_id)
    os.makedirs(obj_indices_save_dir, exist_ok=True)
    pairwise_tokens_save_dir = os.path.join(args.save_dir, 'pairwise_tokens', video_id)
    os.makedirs(pairwise_tokens_save_dir, exist_ok=True)
    perm_save_dir = os.path.join(args.save_dir, 'permutation', video_id)
    os.makedirs(perm_save_dir, exist_ok=True)
    # Loop through video frames and extract HOIs
    filenames = sorted(os.listdir(args.frames_dir))
    for filename in tqdm(filenames):
        frame_id = filename.split(sep='.')[0]
        logits_filename = os.path.join(logits_save_dir, frame_id + '.npy')
        human_indices_filename = os.path.join(human_indices_save_dir, frame_id + '.npy')
        obj_indices_filename = os.path.join(obj_indices_save_dir, frame_id + '.npy')
        pw_tokens_filename = os.path.join(pairwise_tokens_save_dir, frame_id + '.npy')
        perm_filename = os.path.join(perm_save_dir, frame_id + '.npy')
        frame_already_processed = (os.path.isfile(logits_filename) and
                                   os.path.isfile(human_indices_filename) and
                                   os.path.isfile(obj_indices_filename) and
                                   os.path.isfile(pw_tokens_filename) and
                                   os.path.isfile(perm_filename)
                                   )
        if frame_already_processed:
            continue
        filepath = os.path.join(args.frames_dir, filename)
        im = dataset.dataset.load_image(filepath)
        im_tensor, _ = dataset.transforms(im, None)
        im_tensor = im_tensor.to(device)
        pred_hs = load_tensor(os.path.join(features_dir, frame_id + '.npy'), device=device)
        pred_logits = load_tensor(os.path.join(cls_logits_dir, frame_id + '.npy'), device=device)
        pred_logits = pred_logits[:, :, :, KEEP]
        pred_bbs = load_tensor(os.path.join(bbs_dir, frame_id + '.npy'), device=device)
        pred_bbs = rescale_bb_from_original_to_zero_one(pred_bbs, im.width, im.height)
        pred_bbs = xyxy_to_cxcywh(pred_bbs)
        detector_cache = {
            'pred_logits': pred_logits,
            'pred_bbs': pred_bbs,
            'hs': pred_hs,
        }
        with torch.no_grad():
            output = upt([im_tensor], detector_cache=detector_cache, select_top_scoring_human=True)
        np.save(logits_filename, arr=output['logits'].to('cpu').numpy())
        np.save(human_indices_filename, arr=output['human_indices'].to('cpu').numpy())
        np.save(obj_indices_filename, arr=output['object_indices'].to('cpu').numpy())
        np.save(pw_tokens_filename, arr=output['pairwise_tokens'].to('cpu').numpy())
        np.save(perm_filename, arr=output['permutation'].to('cpu').numpy())


if __name__ == '__main__':
    main()
