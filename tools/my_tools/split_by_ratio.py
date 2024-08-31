from PIL import Image
import os
import numpy as np
import argparse
from tqdm import tqdm
import shutil as file
from label_check import *

def get_args_parser():
    parser = argparse.ArgumentParser('split image data into selected ratio', add_help=False)
    parser.add_argument('--data_path', type=str, help='data path ex).../wta_512_aug_lora', required=True)
    parser.add_argument('--show_count', type=bool, default=False, help='show status according to number of defects and normal numbers')
    parser.add_argument('--normal_count', type=int, help='count of normal image')
    parser.add_argument('--defect_count', type=int, help='count of defect image')
    parser.add_argument('--out_path', type=str, default="/shared/home/vclp/hyunwook/junhyung/mmsegmentation/data" ,help='output path')
    return parser.parse_args()

def save_data(args, normal_list, defect_list):
    #저장될 새 디렉토리 생성, 하위 디렉토리 생성
    dir_name = os.path.basename(args.data_path)+"_ratio_"+str(round(len(normal_list)/len(defect_list), 2))
    main_dir_path = os.path.join(args.out_path, dir_name)
    ann_dir_path = os.path.join(main_dir_path, "ann_dir")
    img_dir_path = os.path.join(main_dir_path, "img_dir")
    print("Make new directory...")
    os.mkdir(main_dir_path)
    os.mkdir(ann_dir_path)
    os.mkdir(img_dir_path)
    print("completed.")
    #test, val 파일 복사
    print("Copying except train directory...")
    ori_ann_dir_path = os.path.join(args.data_path, "ann_dir")
    ori_img_dir_path = os.path.join(args.data_path, "img_dir")
    file.copytree(ori_ann_dir_path+"/test", ann_dir_path+"/test")
    file.copytree(ori_img_dir_path+"/test", img_dir_path+"/test")
    file.copytree(ori_ann_dir_path+"/val", ann_dir_path+"/val")
    file.copytree(ori_img_dir_path+"/val", img_dir_path+"/val")
    print("completed.")
    #비율에 맞게 선별된 이미지를 저장
    print("Making train data by ratio...")
    ann_train_path = os.path.join(ann_dir_path, "train")
    img_train_path = os.path.join(img_dir_path, "train")
    ori_ann_train_path = os.path.join(ori_ann_dir_path, "train")
    ori_img_train_path = os.path.join(ori_img_dir_path, "train")
    os.mkdir(ann_train_path)
    os.mkdir(img_train_path)
    
    for normal in tqdm(normal_list, desc="copying filtered normal data", total=len(normal_list)):
        file.copy(ori_ann_train_path+"/"+normal, ann_train_path+"/"+normal)
        file.copy(ori_img_train_path+"/"+normal, img_train_path+"/"+normal)
    for defect in tqdm(defect_list, desc="copying filtered defect data", total=len(defect_list)):
        file.copy(ori_ann_train_path+"/"+defect, ann_train_path+"/"+defect)
        file.copy(ori_img_train_path+"/"+defect, img_train_path+"/"+defect)

    print("completed.")



def check_one_label(img_path): #normal인 경우, 0, defect인 경우, 1 반환
    img = np.array(Image.open(img_path).convert("RGB"))
    unique_pixel = np.unique(img)
    if len(unique_pixel) == 1:
        return 0
    else:
        # print(unique_pixel)
        return 1

def check_possible(normal_count, defect_count):
    label_dir = os.path.join(args.data_path, "ann_dir")
    train_label_dir_path = os.path.join(label_dir, "train")
    sum = 0
    normal = 0
    defect = 0
    for image in os.listdir(train_label_dir_path):
        img_path = os.path.join(train_label_dir_path, image)
        result = check_one_label(img_path)
        if result == 0:
            normal+=1
        else:
            defect+=1
        sum+=1
    print(f"==={dir}===")
    print('normal : ', normal, "chosen normal count : ", normal_count)
    print('defect', defect, "chosen defect count : ", defect_count)
    print('sum : ', sum)
    print(f"===============")

    if normal >= normal_count and defect >= defect_count:
        print("Split your data by ratio is possible!")
        return True
    else:
        return False


def main(args):
    print("[", os.path.basename(args.data_path), "]")
    print("Split ratio > 1 : ", round(args.normal_count/args.defect_count, 2))
    print("[Checing possibility]")
    if check_possible(args.normal_count, args.defect_count):
        print("completed.")
        normal = 0
        defect = 0
        normal_list = []
        defect_list = []
        img_dir_train_path = os.path.join(os.path.join(args.data_path, "ann_dir"), "train") #train 영역만 분리합니다!
        print("[Filtering train data]")
        for image in tqdm(os.listdir(img_dir_train_path), desc="Collecting the image", total=len(os.listdir(img_dir_train_path))):
            if normal == args.normal_count and defect == args.defect_count: #early stopping if possible
                break
            img_path = os.path.join(img_dir_train_path, image)
            result = check_one_label(img_path)
            if result == 0:
                if normal < args.normal_count:
                    normal_list.append(image)
                    normal+=1
            else:
                if defect < args.defect_count:
                    defect_list.append(image)
                    defect+=1
        if normal != args.normal_count or defect != args.defect_count:
            print("Error : Does not match the filtered data count and chosen data count")
            print("[Filtered data count]")
            print("Normal :", normal, "Defect :", defect)
            print("[Chosen data count]")
            print("Normal :", args.normal_count, "Defect :", args.defect_count)
            exit(0)
        print("completed.")
        print("[Saving data]")
        save_data(args, normal_list, defect_list)
        print("completed.")
    else:
        print("Error : it's not possible to make train data by chosen ratio")
        return 0

if __name__ == "__main__":
    args = get_args_parser()
    if args.show_count == True:
        check_label(args.data_path+"/ann_dir")
    else:
        if args.normal_count == None or args.defect_count == None:
            print("--normal_count, --defect_count arguments are required.")
            exit(0)
        else:
            main(args)
