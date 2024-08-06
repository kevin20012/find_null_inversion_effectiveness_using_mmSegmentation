# Comparing using Original dataset and Augmented dataset(by Null-inversion diffusion) with mmSegmentaion
## 학습 및 추론
**bash-run_main.sh** 를 이용해 여러개의 모델을 한번에 학습 및 추론시켜 결과를 뽑아낼 수 있다.  
다음은 run_main.sh를 이용하기 위해 설정해야하는 변수입니다.
``` python
#work_dirs 내의 저장되는 디렉토리의 이름입니다. 따라서 프로젝트 별로 결과를 저장할 수 있습니다.
PROJECT_NAME=train1 

CUDA_VISIBLE_DEVICES=0 

#GPU개수 입력
GPU_COUNT=2 

#시도해볼 모델 - configs 디렉토리 내의 py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
# 예시) configs/deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta-256x256 인 경우, 
# => deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta-256x256

MODEL=( 
    deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta-256x256\ 
    bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta-256x256\
    fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta-256x256
)
#augmentation config를 이용해 시도해볼 모델 - config py파일의 상위 디렉토리의 이름도 함께 써주어야합니다.
MODEL_AUG=( 
    deeplabv3/deeplabv3_r50-d8_4xb4-40k_wta_aug-256x256\ 
    bisenetv1/bisenetv1_r18-d32_4xb4-40k_wta_aug-256x256\
    fastfcn/fastfcn_r50-d32_jpu_psp_4xb4-40k_wta_aug-256x256
)

#전반적인 config를 모두 담고있는 configs파일의 위치
MODEL_CONFIG_PATH=/mmsegmentation/configs

#결과가 저장될 디렉토리를 입력
WORK_DIR=/mmsegmentation/work_dirs
```
**파일 저장 구조**
```
./workdirs/[Project Name]
                ├── aug
                │   ├── [model디렉토리/model명]
                │   │           ├─────── [실행시간으로 정의된 디렉토리]
                │   │           │                    ├── vis_data
                │   │           │                    │       ├── [실행시간으로 정의된 이름].json
                │   │           │                    │       ├── config.py
                │   │           │                    │       └── scalars.json
                │   │           │                    └── [실행시간으로 정의된 이름].log
                │   │           └─────── ckpt (가중치 파일)
                │   │
                │   │
                │   ├── [model디렉토리/model명]         
                │   .                                
                │   .                                
                │   .
                └── origin
                    ├── [model디렉토리/model명]
                    ├── [model디렉토리/model명]
                    .
                    .
```
**실행 결과를 종합해서 표로 뽑고 싶다면...**  
**make_csv.py** 를 이용해보세요!
```bash
make_csv.py --json_path [vis_data내의 [실행시간으로 정의된 이름].json] --defect_matric_path [[실행시간으로 정의된 디렉토리] 내의 [실행시간으로 정의된 이름].log]
```
각 Log 파일들로부터 metric결과를 종합해 result.xlsx를 반환해줍니다.

## mmSegmentation의 간단한 설명
### 구조
기본적으로 train.py를 살펴보면, configs 디렉토리 내의 config 파일 하나만을 가지고 학습을 진행하는 것을 볼 수 있습니다.  
이 config파일안에 데이터셋, 모델 등 모든 내용이 들어있습니다. 그럼 코드를 보면서 진행해보겠습니다.
#### Config 파일
```python
# _base__ 리스트를 통해 
_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', #어떤 모델을 사용할건지,
    '../_base_/datasets/wta_512.py', #어떤 데이터셋 옵션을 사용할건지
    '../_base_/default_runtime.py', #(이건 기본 스케줄 옵션입니다.)
    '../_base_/schedules/schedule_40k.py' #얼마나 반복할지에 대한 정보를 가져올 수 있습니다.
]
# 아래에 작성된 것은 기존 모델의 구조를 변경하고 싶을때 사용합니다. 기존 모델의 코드에 아래코드가 오버라이딩됩니다.
# 따라서 이용하고 싶은 데이터셋에 맞도록, crop_size, num_classes를 변경하는 것이 중요합니다.
crop_size = (256, 256)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=3),
    auxiliary_head=dict(num_classes=3))

```
#### Dataset config 파일
configs/_base_/datasets 디렉토리 내에 위치합니다.  
해당 config 파일은 데이터셋에 대한 옵션을 담고 있습니다.  
```python
dataset_type = 'WTADataset' #어떤 타입의 데이터셋인지
data_root = './data/wta_512/' #데이터의 위치는 어디인지
...
data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train') #wta512/img_dir: 이미지가 들어있음. wta512/ann_dir에는 레이블이 들어있음을 알려줌.
...
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU']) #평가 방법을 정함.
test_evaluator = val_evaluator
```

### 새로운 데이터셋을 사용하고 싶다면  
mmseg/datasets 디렉토리 내에 새로운 데이터에 대한 정보를 담은 클래스를 만들어주어야합니다.  
```python
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module() #중요
class WTADataset(BaseSegDataset):
    METAINFO = dict(
        classes=('defect', 'attached', 'broken'), # 사용하고 싶은 데이터의 레이블이름
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]]) # 각 레이블에 대응되는 색상

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=False,
            **kwargs)
```  
그리고 가장 중요한 것이 하나 남았습니다.
```
mmseg/datasets/__init__.py
```
에 해당 클래스가 있음을 알려줘야합니다.  
**__init__.py**를 열고 
```python
from .wta import WTADataset #위에서 작성한 클래스 모듈을 import
__all__ = [..., 'WTADataset'] #클래스 이름을 이렇게 __all__에 추가해줍니다.
```
그리고 위에서 서술한 dataset_config 파일을 작성한뒤, config파일의 _base_ 리스트 내에 이를 추가해서 사용할 수 있습니다!  
만약 새로 추가한 데이터셋 모듈이 없다고하는 에러가 나게되면, recompile과정이 이 에러를 해결해줄 수 있습니다.  
```bash
pip install -v -e . # mmsegmentation 디렉토리 내에서 실행
```
에러가 해결될 것입니다!