import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from PIL import Image


COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
COCO_CLASSES = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]


def load_hico_det_verbs(filepath):
    verb_names = []
    with open(filepath, mode='r') as f:
        next(f)
        next(f)
        for line in f:
            _, verb_name = line.strip().split()
            verb_names.append(verb_name.replace('_', '-'))
    return verb_names


def plot_bbs(pil_img, boxes, bbs_prob):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(bbs_prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        text = f'{COCO_CLASSES[cl]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def plot_results(pil_img, boxes, hoi_prob, hoi_classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    if len(boxes) > 0:
        xmin, ymin, xmax, ymax = boxes[0].tolist()
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=COLORS[-1], linewidth=3))
        text = 'human'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
        for p, (xmin, ymin, xmax, ymax), c in zip(hoi_prob, boxes[1:].tolist(), COLORS * 100):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{hoi_classes[cl]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


def select_bounding_boxes_to_visualise(bbs_logits, threshold):
    probas = softmax(bbs_logits, axis=-1)[..., :-1]
    keep = np.max(probas, axis=-1) > threshold
    if len(keep) > 0:
        keep[0] = True
    return keep


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames_dir', type=str, required=True,
                        help='Directory containing video frames.')
    parser.add_argument('--object_detections_dir', type=str, required=True,
                        help='Directory containing all detections. E.g. '
                             '/home/romero/data/MPIICooking2/object-detection/detr')
    parser.add_argument('--hoi_detections_dir', type=str, required=True,
                        help='Directory containing all HOI detections. E.g. '
                             '/home/romero/data/MPIICooking2/human-object-interaction')
    return parser


def main():
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args()
    video_id = os.path.basename(args.frames_dir)
    # Directories containing needed data for visualisation
    bbs_dir = os.path.join(args.object_detections_dir, 'bounding_boxes', video_id)
    bbs_logits_dir = os.path.join(args.object_detections_dir, 'classes_logits', video_id)
    hoi_logits_dir = os.path.join(args.hoi_detections_dir, 'classes_logits', video_id)
    perm_dir = os.path.join(args.hoi_detections_dir, 'permutation', video_id)
    # Loop through video frames and visualise HOIs
    hoi_classes = load_hico_det_verbs('/home/romero/data/HICO-DET/hico_list_vb.txt')
    filenames = sorted(os.listdir(args.frames_dir))
    for filename in filenames[::50]:
        frame_id = filename.split(sep='.')[0]
        perm = np.load(os.path.join(perm_dir, frame_id + '.npy'))
        bbs_logits = np.load(os.path.join(bbs_logits_dir, frame_id + '.npy'))[perm]
        keep = select_bounding_boxes_to_visualise(bbs_logits, threshold=0.3)
        bbs = np.load(os.path.join(bbs_dir, frame_id + '.npy'))[perm][keep]
        hoi_logits = np.load(os.path.join(hoi_logits_dir, frame_id + '.npy'))[keep[1:]]
        hoi_prob = softmax(hoi_logits, axis=-1)
        img = Image.open(os.path.join(args.frames_dir, filename))
        plot_results(img, boxes=bbs, hoi_prob=hoi_prob, hoi_classes=hoi_classes)


if __name__ == '__main__':
    main()
