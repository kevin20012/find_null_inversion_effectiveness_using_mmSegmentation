from PIL import Image
import os
import numpy as np
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('check_image_label', add_help=False)
    parser.add_argument('--label_dir', type=str, help='path to ann_dir', required=True)
    parser.add_argument('--save_dir', type=str, help='path to save dir', required=True)
    return parser.parse_args()

def main(args):
    os.mkdir(args.save_dir)
    for dir in os.listdir(args.label_dir):
        dir_path = os.path.join(args.label_dir, dir)
        save_path = os.path.join(args.save_dir, dir)
        os.mkdir(save_path)
        for image_name in os.listdir(dir_path):
            img_path = os.path.join(dir_path, image_name)
            img = np.array(Image.open(img_path).convert("RGB"))
            find0, find1, find2 = 0, 0, 0
            for c in range(img.shape[0]):
                for r in range(img.shape[1]):
                    if np.unique(img[c][r])[0] == 0:
                        find0+=1
                        pass
                    elif np.unique(img[c][r])[0] == 1:
                        find1+=1
                        img[c][r] = np.array([255, 0, 0])
                    elif np.unique(img[c][r])[0] == 2:
                        find2+=1
                        img[c][r] = np.array([0, 0, 255])
                    
            img = Image.fromarray(img).save(os.path.join(save_path, image_name), 'png')
            print(f"{image_name}")
            print(f'result : {find0}, {find1}, {find2}')
            

if __name__ == "__main__":
    args = get_args_parser()

    main(args)