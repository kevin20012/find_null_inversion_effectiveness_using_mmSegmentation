#! /bin/bash
PROJECT_NAME=train1 #work_dirs 내의 저장되는 디렉토리의 이름입니다.
CUDA_VISIBLE_DEVICES=0 
GPU_COUNT=2 #GPU개수 입력
MODEL=( #시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
    deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta-256x256\ 
    bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta-256x256\
    fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta-256x256
)
MODEL_AUG=( #aug 시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
    deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta_aug-256x256\ 
    bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta_aug-256x256\
    fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta_aug-256x256
)
MODEL_CONFIG_PATH=/shared/home/vclp/hyunwook/junhyung/mmsegmentation/configs
WORK_DIR=/shared/home/vclp/hyunwook/junhyung/mmsegmentation/work_dirs

echo "🔥🔥🔥🔥🔥start training and testing🔥🔥🔥🔥🔥"
mkdir -p $WORK_DIR/$PROJECT_NAME
touch $WORK_DIR/$PROJECT_NAME/log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/total_log.txt


# 원본 학습 후 추론 진행
mkdir -p $WORK_DIR/$PROJECT_NAME/origin
touch $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
for model in ${MODEL[@]}
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
    mkdir -p $WORK_DIR/$PROJECT_NAME/origin/$model/test
    python tools/test.py \
    $MODEL_CONFIG_PATH/$model.py $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt/`basename $(< $WORK_DIR/$PROJECT_NAME/origin/$model/train/ckpt/last_checkpoint)` --out $WORK_DIR/$PROJECT_NAME/origin/$model/test > $WORK_DIR/$PROJECT_NAME/origin/$model/test/test_log.txt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Test End=================" >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
    echo " " >> $WORK_DIR/$PROJECT_NAME/origin/origin_log.txt
done

#증강 데이터 학습 후 추론 진행

mkdir -p $WORK_DIR/$PROJECT_NAME/aug
touch $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
echo "$(date +%Y-%m-%d-%H:%M:%S) <<  ${PROJECT_NAME}  >>" > $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
for model in ${MODEL_AUG[@]}
do
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================$model=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    echo "\n$(date +%Y-%m-%d-%H:%M:%S) 🚀🚀🚀🚀🚀$model Start!!!🚀🚀🚀🚀🚀"

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Train Start=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    # 학습 시작
    bash tools/dist_train.sh \
    $MODEL_CONFIG_PATH/$model.py \
    $GPU_COUNT --work-dir $WORK_DIR/$PROJECT_NAME/aug/$model/train
    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Train End=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt


    #가중치 파일 이동
    mkdir -p $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt
    mv $WORK_DIR/$PROJECT_NAME/aug/$model/train/iter_* $WORK_DIR/$PROJECT_NAME/aug/$model/train/last_checkpoint $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Test Start=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    # 추론 시작
    mkdir -p $WORK_DIR/$PROJECT_NAME/aug/$model/test
    python tools/test.py \
    $MODEL_CONFIG_PATH/$model.py $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt/`basename $(< $WORK_DIR/$PROJECT_NAME/aug/$model/train/ckpt/last_checkpoint)` --out $WORK_DIR/$PROJECT_NAME/aug/$model/test > $WORK_DIR/$PROJECT_NAME/aug/$model/test/test_log.txt

    echo "$(date +%Y-%m-%d-%H:%M:%S) =================Original Test End=================" >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
    echo " " >> $WORK_DIR/$PROJECT_NAME/aug/aug_log.txt
done
