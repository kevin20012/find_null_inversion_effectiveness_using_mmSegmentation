import pandas as pd
import os
import argparse
from ast import literal_eval


#json 파일을 입력으로 받아 해당 파일을 요약해서 csv 파일을 만들어줍니다.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str, help='json, log 파일 2개가 존재하는 위치를 입력하면 아래 옵션을 입력하지 않고 자동으로 만들어줍니다.')
    parser.add_argument('--json_path', type=str, help="json 로그 파일의 위치")
    parser.add_argument('--defect_matric_path', type=str, help="각 결함별 metric이 나와있는 Log 파일위치")
    parser.add_argument('--out_path', type=str, help="결과 파일 저장 위치")

    return parser.parse_args()

    

def main():
    args = parse_args()
    if args.log_path != None:
        for item in os.listdir(args.log_path):
            if item.find(".json") != -1:
                print("find json file!")
                json_path = os.path.join(args.log_path, item)
            elif item.find(".log") != -1:
                print("find log file!")
                metric_path = os.path.join(args.log_path, item)
            else:
                pass
    else:
        json_path = args.json_path
        metric_path = args.defect_matric_path
    if args.out_path != None:
        out_path = args.out_path
    else:
        out_path = args.log_path
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

