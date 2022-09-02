_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/wider_face_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    rpn_head=dict(
        anchor_generator=dict(scales=[4])
    )
)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data_root = 'dataset/WIDERFace/'
data = dict(
    train=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_train_70.json',
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
    ),
    test=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
        pipeline=test_pipeline
    )
)
# batch size factor
# bs_factor = 2
# optimizer
# optimizer = dict(type='SGD', lr=0.012/bs_factor, momentum=0.9, weight_decay=5e-4)
# optimizer_config = dict()
# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=1000,
#     warmup_ratio=0.001,
#     step=[16, 20])
# runtime settings
# runner = dict(type='EpochBasedRunner', max_epochs=24)
# log_config = dict(interval=1)

auto_scale_lr = dict(enable=True)