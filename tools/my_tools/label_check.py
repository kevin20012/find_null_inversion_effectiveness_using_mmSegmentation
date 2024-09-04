from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm

def get_args_parser():
    parser = argparse.ArgumentParser('check_image_label', add_help=False)
    parser.add_argument('--label_dir', type=str, help='path to data dir', required=True)
    return parser.parse_args()

def check_label(ann_dir):
    sum = {}
    normal = {}
    defect = {}
    for dir in os.listdir(ann_dir):
        dir_path = os.path.join(ann_dir, dir)
        sum[dir] = 0
        normal[dir] = 0
        defect[dir] = 0
        for image in tqdm(os.listdir(dir_path), desc="["+dir+"] checking", total=len(os.listdir(dir_path))):
            img_path = os.path.join(dir_path, image)
            img = np.array(Image.open(img_path).convert("RGB"))
            # for i in img[:][:]:
            #     for j in i:
            #         flag = 0
            #         for n in j:
            #             if n!=0:
            #                 flag=1
            #                 break
            #         if flag == 1:
            #             print(j)
            # break
            unique_pixel = np.unique(img)
            if len(unique_pixel) == 1:
                normal[dir]+=1
            else:
                # print(unique_pixel)
                defect[dir]+=1
            sum[dir]+=1

    for dir in os.listdir(ann_dir):
        print(f"==={dir}===")
        print('normal', normal[dir])
        print('defect', defect[dir])
        print('sum : ', sum[dir])
        print('ratio : 1 :', round(normal[dir]/defect[dir], 2))

if __name__ == "__main__":
    args = get_args_parser()

    check_label(args.label_dir+"/ann_dir")
