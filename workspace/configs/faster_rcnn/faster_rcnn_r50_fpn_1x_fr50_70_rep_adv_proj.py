_base_ = './faster_rcnn_r50_fpn_1x_proj.py'

dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
CLASSES = ('person','car','bus','train','truck','camouflage man', 'cruiser/destroyer', 'littoral combat ship','landing helicopte ship','aircraft carrier','fighter', 'patrol aircraft','uav', 'bomber' )

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

adv_train = dict(
    type=dataset_type,
    classes = CLASSES,
    img_prefix=data_root + 'ssd512_8_5',
    ann_file=data_root+'labels/psd_fr50_ssd512_adv_8_5_train_70.json',
    pipeline=train_pipeline
)
ori_train = dict(
    type=dataset_type,
    classes = CLASSES,
    img_prefix=data_root + 'images',
    ann_file=data_root+'labels/psd_fr50_train_70.json',
    pipeline=train_pipeline
)

data = dict(

    train = [ 
            dict(
                _delete_=True,
                type='RepeatDataset',
                times=2,
                dataset=ori_train), 
            adv_train
        ],
    val=dict(

        ann_file=data_root+'labels/psd_fr50_test_70.json',


    ),
    test=dict(
 
        ann_file=data_root+'labels/psd_fr50_test_70.json',


    )
)

