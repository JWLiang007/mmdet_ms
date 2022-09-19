_base_ = './yolov3_d53_mstrain-608_273e_coco.py'
# model settings
model = dict(
    bbox_head=dict(
        num_classes=18,
    )
)
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(608, 608), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

CLASSES = ('CG','CVN','DDG','LCS','LHA', 'LHD','B52', 'F16', 'F22','F35', 'P8A','RQ4' , 'person','car','bus','train', 'truck','camouflage man')
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/train_all.json',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json',
        ),
    test=dict(
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json',
        ))

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True,base_batch_size=64)

lr_config = dict(
    step=[109, 123])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=135)