import pandas as pd
import os
import argparse
from ast import literal_eval


#json 파일을 입력으로 받아 해당 파일을 요약해서 csv 파일을 만들어줍니다.
# 필요 파일 : train 폴더 내에 [vis_data 디렉토리 내의 json파일], [log파일] 2개입니다.
# 각각을 --json_path, --defect_matric_path 에 해당 파일의 경로를 인수로 주면 작동합니다.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, help="json 로그 파일의 위치")
    parser.add_argument('--defect_matric_path', type=str, help="각 결함별 metric이 나와있는 Log 파일위치")
    parser.add_argument('--out_path', default='./', help="결과 파일 저장 위치")

    return parser.parse_args()

    

def main():
    args = parse_args()
    json_path = args.json_path
    metric_path = args.defect_matric_path
    out_path = args.out_path
    loss = []
    step = []
    miou = []
    aAcc = []
    mAcc = []
    defect_iou = []
    defect_acc = []
    attached_iou = []
    attached_acc = []
    broken_iou = []
    broken_acc = []
    result = pd.DataFrame(columns=['step', 'loss', 'miou', 'aAcc', 'mAcc', 'defect_iou', 'defect_acc','attached_iou', 'attached_acc','broken_iou', 'broken_acc'])

    f = open(json_path, 'r')
    text = f.read()
    text = text.split("\n")

    for line in text:
        try:
            dic = literal_eval(line)
            if 'mIoU' not in dic.keys():
                loss.append(dic["loss"])
                step.append(dic["step"])
                miou.append("")
                aAcc.append("")
                mAcc.append("")
            else:
                miou[-1] = (dic["mIoU"])
                aAcc[-1] = (dic["aAcc"])
                mAcc[-1] = (dic["mAcc"])
        except:
            pass

    f.close()

    result['step']=step
    result['loss']=loss
    result['miou']=miou
    result['aAcc']=aAcc
    result['mAcc']=mAcc

    f = open(metric_path, 'r')
    text = f.read()
    text = text.split('\n')

    step = 0
    for line in text:
        print(line)
        if line.find("Saving checkpoint") > 0:
            step = int(line.replace(' ', '').split('at')[1].split('iter')[0])

        if line.find('defect') > 0 or line.find('attached') > 0 or line.find('broken') > 0:
            class_and_points = (line.replace(' ', '')).split('|')[1:-1]
            if class_and_points[0] == 'defect':
                defect_iou.append((step, class_and_points[1]))
                defect_acc.append((step, class_and_points[2]))
            elif class_and_points[0] == 'attached':
                attached_iou.append((step, class_and_points[1]))
                attached_acc.append((step, class_and_points[2]))
            elif class_and_points[0] == 'broken':
                broken_iou.append((step, class_and_points[1]))
                broken_acc.append((step, class_and_points[2]))
    f.close()

    for i in range(len(defect_acc)):
        result.loc[result['step']==defect_iou[i][0], 'defect_iou'] = float(defect_iou[i][1])
        result.loc[result['step']==defect_acc[i][0], 'defect_acc'] = float(defect_acc[i][1])
        result.loc[result['step']==attached_iou[i][0], 'attached_iou'] = float(attached_iou[i][1])
        result.loc[result['step']==attached_acc[i][0], 'attached_acc'] = float(attached_acc[i][1])
        result.loc[result['step']==broken_iou[i][0], 'broken_iou'] = float(broken_iou[i][1])
        result.loc[result['step']==broken_acc[i][0], 'broken_acc'] = float(broken_acc[i][1])

    result.to_excel(os.path.join(out_path, 'result.xlsx'))

if __name__ == "__main__":
    main()

