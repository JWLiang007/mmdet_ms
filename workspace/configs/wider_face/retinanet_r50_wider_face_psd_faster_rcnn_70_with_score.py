_base_ = [
    './retinanet_r50_wider_face.py'
]



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
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels','scores']),
]

data_root = 'dataset/WIDERFace/'
data = dict(
    train=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_train_70_with_score.json',
        pipeline=train_pipeline
    ),
    val=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
    ),
    test=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
    )
)