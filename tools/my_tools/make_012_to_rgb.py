from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm

BLENDING_ALPHA = 0.5

def get_args_parser():
    parser = argparse.ArgumentParser('check_image_label', add_help=False)
    parser.add_argument('--data_dir', type=str, help='path to data dir ex. ~/wta_512', required=True)
    parser.add_argument('--save_dir', type=str, help='path to save dir', required=True)
    return parser.parse_args()

def image_bleding(image: np.array, label:np.array)->np.array:
    return (1-BLENDING_ALPHA)*image + BLENDING_ALPHA*label

def main(args):
    save_dir = os.path.join(args.save_dir, args.data_dir.split('/')[-1]+'_to_rgb')
    os.mkdir(save_dir)
    ann_dir = os.path.join(args.data_dir, 'ann_dir')
    img_dir = os.path.join(args.data_dir, 'img_dir')

    for dir in os.listdir(ann_dir):
        print(f"{dir} 진행 중...")
        label_path = os.path.join(ann_dir, dir)
        img_path = os.path.join(img_dir, dir)
        save_path = os.path.join(save_dir, dir)
        os.mkdir(save_path)
        for image_name in tqdm(os.listdir(label_path)):
            label_image = os.path.join(label_path, image_name)
            img_image = os.path.join(img_path, image_name)
            label = np.array(Image.open(label_image).convert("RGB"))
            img = np.array(Image.open(img_image).convert("RGB"))
            find0, find1, find2 = 0, 0, 0
            for c in range(label.shape[0]):
                for r in range(label.shape[1]):
                    if np.unique(label[c][r])[0] == 0:
                        find0+=1
                        pass
                    elif np.unique(label[c][r])[0] == 1:
                        find1+=1
                        label[c][r] = np.array([255, 0, 0])
                    elif np.unique(label[c][r])[0] == 2:
                        find2+=1
                        label[c][r] = np.array([0, 0, 255])

            blending = image_bleding(img, label)
            result = np.concatenate((img, label, blending), axis=1).astype(np.uint8)       
            result = Image.fromarray(result).save(os.path.join(save_path, image_name), 'png')
            

if __name__ == "__main__":
    args = get_args_parser()

    main(args)