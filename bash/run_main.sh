#! /bin/bash
read -p "Enter Project Name : " PROJECT_NAME #work_dirs 내의 저장되는 디렉토리의 이름입니다.
if [ "$PROJECT_NAME" == "" ]; then
    echo "Empty name is not acceptable."
    exit 1
fi
CUDA_VISIBLE_DEVICES=0 
read -p "Enter Gpu Count : " GPU_COUNT #GPU개수 입력
if (($GPU_COUNT <= 0)); then
    echo "0 or Negative number of Gpu is not acceptable."
    exit 1
fi
GROUP_1=( #시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
    # 처음 비교
    # deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta-256x256\ 
    # bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta-256x256\
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta-256x256

    # crop_size에 따른 비교
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta-512x512\
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta-1024x1024

    #결함 : 정상 비율에 따른 비교
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta-512x512\
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta_ratio_10-512x512

    #CNN 기반 모델 - FCN, U-Net, Deeplabv3+, PSPNet
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta-512x512\
    # pspnet/pspnet_r50-d8_4xb4-40k_wta512-512x512\
    #트랜스포머 기반 모델 - SETR, Swin, Segmenter, Mask2Former
    # setr/setr_vit-l_naive_8xb1-40k_wta512-512x512\
    # segformer/segformer_mit-b0_8xb2-40k_wta512-512x512

)
GROUP_2=( #aug 시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
    # 처음 비교
    # deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta_aug-256x256\ 
    # bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta_aug-256x256\
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta_aug-256x256

    # crop_size에 따른 비교
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta_aug-512x512\
    # fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta_aug-1024x1024

    #결함 : 정상 비율에 따른 비교
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta_aug_wolora-512x512\
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta_aug_wolora_ratio_10-512x512

    #CNN 기반 모델 - FCN, U-Net, Deeplabv3+, PSPNet
    # deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta_aug_wolora-512x512\
    # pspnet/pspnet_r50-d8_4xb4-40k_wta512_aug-512x512\
    #트랜스포머 기반 모델 - SETR, Swin, Segmenter, Mask2Former
    # setr/setr_vit-l_naive_8xb1-40k_wta512_aug-512x512\
    # segformer/segformer_mit-b0_8xb2-40k_wta512_aug-512x512

    #relabeling 한 데이터 실험
    deeplabv3plus/deeplabv3plus_r18-d8_4xb2-40k_wta_aug_relabeling-512x512\
    pspnet/pspnet_r50-d8_4xb4-40k_wta512_aug_relabeling-512x512\
    setr/setr_vit-l_naive_8xb1-40k_wta512_aug_relabeling-512x512\
    segformer/segformer_mit-b0_8xb2-40k_wta512_aug_relabeling-512x512
)
MODEL_CONFIG_PATH=/shared/home/vclp/hyunwook/junhyung/mmsegmentation/configs
WORK_DIR=/shared/home/vclp/hyunwook/junhyung/mmsegmentation/work_dirs
FIND_BEST_DIR=/shared/home/vclp/hyunwook/junhyung/mmsegmentation/bash
METRIC=mIoU

echo "🔥🔥🔥🔥🔥start training and testing🔥🔥🔥🔥🔥"
mkdir -p $WORK_DIR/$PROJECT_NAME
touch $WORK_DIR/$PROJECT_NAME/log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/total_log.txt


# 원본 학습 후 추론 진행
mkdir -p $WORK_DIR/$PROJECT_NAME/origin
touch $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
for model in ${GROUP_1[@]}
do
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================$model=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
    echo "\n$(date +%Y-%m-%d-%H:%M:%S) 🚀🚀🚀🚀🚀$model Start!!!🚀🚀🚀🚀🚀"

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Train Start=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
    # 학습 시작
    bash tools/dist_train.sh \
    $MODEL_CONFIG_PATH/$model.py \
    $GPU_COUNT --work-dir $WORK_DIR/$PROJECT_NAME/origin/$model/train
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Train End=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt


    #가중치 파일 이동
    mkdir -p $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt
    mv $WORK_DIR/$PROJECT_NAME/origin/$model/train/iter_* $WORK_DIR/$PROJECT_NAME/origin/$model/train/last_checkpoint $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Test Start=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
    # 추론 시작
    vis_data_path=$WORK_DIR/$PROJECT_NAME/origin/$model/train/$(ls $WORK_DIR/$PROJECT_NAME/origin/$model/train | grep [0-9][0-9][0-9]_[0-9][0-9][0-9])/vis_data
    json_log_path=$vis_data_path/$(ls $vis_data_path | grep [0-9]*_[0-9]*.json)
    #최고 점수를 가지는 가중치를 선택해 test 진행
    echo iter_`python $FIND_BEST_DIR/find_best.py --json_log_path $json_log_path`.pth > $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt/best_checkpoint
    mkdir -p $WORK_DIR/$PROJECT_NAME/origin/$model/test
    python tools/test.py \
    $MODEL_CONFIG_PATH/$model.py $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt/$(< $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt/best_checkpoint) --out $WORK_DIR/$PROJECT_NAME/origin/$model/test > $WORK_DIR/$PROJECT_NAME/origin/$model/test/test_log.txt
    # 결과를 csv로 변형
    temp_dir = $WORK_DIR/$PROJECT_NAME/origin/$model/train/$(ls $WORK_DIR/$PROJECT_NAME/origin/$model/train | grep [0-9][0-9][0-9]_[0-9][0-9][0-9])
    python $FIND_BEST_DIR/make_csv.py --json_path $json_log_path --defect_matric_path  $temp_dir/$(ls $temp_dir | grep [0-9]*_[0-9]*.log) --out_path $WORK_DIR/$PROJECT_NAME/origin/$model
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Test End=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
    echo " " >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
done

#증강 데이터 학습 후 추론 진행

mkdir -p $WORK_DIR/$PROJECT_NAME/aug
touch $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
for model in ${GROUP_2[@]}
do
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================$model=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    echo "\n$(date +%Y-%m-%d-%H:%M:%S) 🚀🚀🚀🚀🚀$model Start!!!🚀🚀🚀🚀🚀"

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Augmentation Train Start=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    # 학습 시작
    bash tools/dist_train.sh \
    $MODEL_CONFIG_PATH/$model.py \
    $GPU_COUNT --work-dir $WORK_DIR/$PROJECT_NAME/aug/$model/train
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Augmentation Train End=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt


    #가중치 파일 이동
    mkdir -p $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt
    mv $WORK_DIR/$PROJECT_NAME/aug/$model/train/iter_* $WORK_DIR/$PROJECT_NAME/aug/$model/train/last_checkpoint $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Augmentation Test Start=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    # 추론 시작
    vis_data_path=$WORK_DIR/$PROJECT_NAME/aug/$model/train/$(ls $WORK_DIR/$PROJECT_NAME/aug/$model/train | grep [0-9][0-9][0-9]_[0-9][0-9][0-9])/vis_data
    json_log_path=$vis_data_path/$(ls $vis_data_path | grep [0-9]*_[0-9]*.json)
    echo iter_`python $FIND_BEST_DIR/find_best.py --json_log_path $json_log_path`.pth > $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt/best_checkpoint
    mkdir -p $WORK_DIR/$PROJECT_NAME/aug/$model/test
    python tools/test.py \
    $MODEL_CONFIG_PATH/$model.py $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt/$(< $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt/best_checkpoint) --out $WORK_DIR/$PROJECT_NAME/aug/$model/test > $WORK_DIR/$PROJECT_NAME/aug/$model/test/test_log.txt
    # 결과를 csv로 변형
    temp_dir = $WORK_DIR/$PROJECT_NAME/aug/$model/train/$(ls $WORK_DIR/$PROJECT_NAME/aug/$model/train | grep [0-9][0-9][0-9]_[0-9][0-9][0-9])
    python $FIND_BEST_DIR/make_csv.py --json_path $json_log_path --defect_matric_path  $temp_dir/$(ls $temp_dir | grep [0-9]*_[0-9]*.log) --out_path $WORK_DIR/$PROJECT_NAME/aug/$model
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Augmentation Test End=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    echo " " >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
done
