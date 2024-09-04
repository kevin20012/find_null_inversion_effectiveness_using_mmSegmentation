from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
import shutil as file

#train 디렉토리만 변합니다.
def get_args_parser():
    parser = argparse.ArgumentParser('make_label_012', add_help=False)
    parser.add_argument('--data_dir', type=str, help='path to data dir ex. ~/wta_512', required=True)
    parser.add_argument('--save_dir', type=str, help='path to save dir', required=True)
    return parser.parse_args()

def main(args):
    save_dir = os.path.join(args.save_dir, args.data_dir.split('/')[-1]+'_to_012')
    os.mkdir(save_dir)
    ann_dir = os.path.join(args.data_dir, 'ann_dir')
    img_dir = os.path.join(args.data_dir, 'img_dir')
    save_ann_dir = os.path.join(save_dir, 'ann_dir')
    save_img_dir = os.path.join(save_dir, 'img_dir')
    os.mkdir(save_ann_dir)
    os.mkdir(save_img_dir)

    for dir in ['train', 'val', 'test']:
        print(f"{dir} 진행 중...")
        label_path = os.path.join(ann_dir, dir)
        img_path = os.path.join(img_dir, dir)

        if dir == "train":
            save_train_label_path = os.path.join(os.path.join(save_dir, 'ann_dir'), 'train')
            save_train_img_path = os.path.join(os.path.join(save_dir, 'img_dir'), 'train')
            os.mkdir(save_train_label_path)
            file.copytree(img_path, save_train_img_path)

            for image_name in tqdm(os.listdir(label_path)):
                label_image = os.path.join(label_path, image_name)
                
                label = np.array(Image.open(label_image).convert("RGB"))
                #gray 스케일로 새로 저장될 레이블 선언
                new_label = np.array([0]*(label.shape[0]*label.shape[1]), dtype=np.uint8)
                new_label = np.reshape(new_label, (label.shape[0], label.shape[1]))

                # black: 0, red: 1, blue: 2
                for c in range(label.shape[0]):
                    for r in range(label.shape[1]):
                        if label[c][r][0] > 200:
                            new_label[c][r] = 1
                        elif label[c][r][2] > 200:
                            new_label[c][r] = 2
                        else:
                            pass 

                Image.fromarray(new_label).convert("L").save(os.path.join(save_train_label_path, image_name), 'png')
        else:
            temp_label_dir = os.path.join(save_ann_dir, dir)
            temp_img_dir = os.path.join(save_img_dir, dir)
            file.copytree(label_path, temp_label_dir)
            file.copytree(img_path, temp_img_dir)

            

if __name__ == "__main__":
    args = get_args_parser()

    main(args)