_base_ = './yolov3_proj.py'

dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
CLASSES = ('person','car','bus','train','truck','camouflage man', 'cruiser/destroyer', 'littoral combat ship','landing helicopte ship','aircraft carrier','fighter', 'patrol aircraft','uav', 'bomber' )
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


adv_train = dict(
    type=dataset_type,
    classes = CLASSES,
    img_prefix=data_root + 'ssd512_8_5',
    ann_file=data_root+'labels/psd_yolov3_ssd512_adv_8_5_train_70.json',
    pipeline=train_pipeline
)
ori_train = dict(
    type=dataset_type,
    classes = CLASSES,
    img_prefix=data_root + 'images',
    ann_file=data_root+'labels/psd_yolov3_train_70.json',
    pipeline=train_pipeline
)

data = dict(
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=ori_train),
    val=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'),
    test=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'))