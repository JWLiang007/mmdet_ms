_base_ = 'ssd512_coco.py'
input_size = 512
model = dict(
    bbox_head=dict(
        num_classes=18,))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
CLASSES = ('CG','CVN','DDG','LCS','LHA', 'LHD','B52', 'F16', 'F22','F35', 'P8A','RQ4' , 'person','car','bus','train', 'truck','camouflage man')
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes = CLASSES,
            img_prefix=data_root + 'images',
            ann_file=data_root+'labels/train_all.json',
            pipeline=train_pipeline
            )),
    val=dict(        
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json'),
    test=dict(        
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json'))


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
