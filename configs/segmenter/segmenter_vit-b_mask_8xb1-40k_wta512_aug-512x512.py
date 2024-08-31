_base_ = [
    '../_base_/models/segmenter_vit-b16_mask.py',
    '../_base_/datasets/wta_512_aug_wolora_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py']

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor,
             decode_head=dict(num_classes=150),
            auxiliary_head=dict(num_classes=150),
             )
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(
    # num_gpus: 8 -> batch_size: 8
    batch_size=1)
val_dataloader = dict(batch_size=1)
