from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
import shutil as file

BLENDING_ALPHA = 0.5

def get_args_parser():
    parser = argparse.ArgumentParser('candidate_image', add_help=False)
    parser.add_argument('--candidate_txt', type=str, help='path to candidate(for labeling) dir', required=True)
    parser.add_argument('--data_dir', type=str, help='path to data dir', required=True)
    parser.add_argument('--output_dir', type=str, help='path to output dir', required=True)
    return parser.parse_args()

def image_bleding(image: np.array, label:np.array)->np.array:
    return (1-BLENDING_ALPHA)*image + BLENDING_ALPHA*label

def main(args):
    output_path = os.path.join(args.output_dir, "new_labeling")
    os.mkdir(output_path)
    data_dir = args.data_dir
    candi_txt = open(args.candidate_txt, 'r')
    candi_item = candi_txt.read().split('\n')
    candi_txt.close()
    # print(candi_item)

    train_ann_dir = os.path.join(data_dir, 'ann_dir', 'train')
    train_img_dir = os.path.join(data_dir, 'img_dir', 'train')

    for item in tqdm(candi_item, desc="saving..."):
        if item+".png" in os.listdir(train_img_dir):
            file.copy(os.path.join(train_img_dir, item+".png"), output_path)
            image = np.array(Image.open(os.path.join(train_img_dir, item+".png")).convert("RGB"))
            label = np.array(Image.open(os.path.join(train_ann_dir, item+".png")).convert("RGB"))
            for c in range(label.shape[0]):
                for r in range(label.shape[1]):
                    if np.unique(label[c][r])[0] == 0:
                        pass
                    elif np.unique(label[c][r])[0] == 1:
                        label[c][r] = np.array([255, 0, 0])
                    elif np.unique(label[c][r])[0] == 2:
                        label[c][r] = np.array([0, 0, 255])
            blending = image_bleding(image, label)
            ori_image_label = np.concatenate((image, label, blending), axis=1).astype(np.uint8)
            Image.fromarray(ori_image_label).save(os.path.join(output_path, "before_"+item+".png"), 'png')
        else:
            print(f"{item} isn't in {args.data_dir}")

if __name__ == "__main__":
    args = get_args_parser()
    main(args)

    
