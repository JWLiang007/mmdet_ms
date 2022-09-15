_base_ = './faster_rcnn_r50_fpn_2x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=18
        )),

)
CLASSES = ('CG','CVN','DDG','LCS','LHA', 'LHD','B52', 'F16', 'F22','F35', 'P8A','RQ4' , 'person','car','bus','train', 'truck','camouflage man')
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=1e-2,
        dataset=dict(  # Dataset_A 的原始配置信息
            type = dataset_type,
            classes = CLASSES,
            img_prefix=data_root + 'images',
            ann_file=data_root+'labels/train_all.json',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root+'labels/val_all.json',
        img_prefix=data_root + 'images',

    ),
    test=dict(
        classes = CLASSES,
        ann_file=data_root+'labels/val_all.json',
        img_prefix=data_root + 'images',

    )
)

auto_scale_lr = dict(enable=True, base_batch_size=16)