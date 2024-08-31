_base_ = [
    '../_base_/datasets/wta_512_512x512.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=3),
    auxiliary_head=dict(in_channels=512, num_classes=3))
