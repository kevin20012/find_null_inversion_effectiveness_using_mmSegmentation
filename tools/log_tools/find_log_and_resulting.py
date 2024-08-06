# train폴더 입력으로 받아, 해당 디렉토리 내에 생긴 디렉토리에 접근해 Log 파일 읽고 요약해주기.

import argparse
import os, sys

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file_path', type=str, help="log파일을 담고있는 디렉토리의 상위 디렉토리 ex. ~/train")
    return parser.parse_args()

def main():
    args = parse_args()
    dir = os.listdir(args.file_path)

    for d in dir:
        if d != 'ckpt':
            under_train_dir = os.path.join(args.file_path, d)
            break
    
    for d in under_train_dir:
        if d.find('log') > 0:
            f = open(os.path.join(under_train_dir,d), 'r')
            break

    #f로 파일 연거 작업해주면 됌.
    