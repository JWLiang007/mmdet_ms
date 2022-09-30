_base_ = 'ssd512_proj.py'

dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
CLASSES = ('person','car','bus','train','truck','camouflage man', 'cruiser/destroyer', 'littoral combat ship','landing helicopte ship','aircraft carrier','fighter', 'patrol aircraft','uav', 'bomber' )

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

adv_train =dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes = CLASSES,
            img_prefix=data_root + 'ssd512_8_5',
            ann_file=data_root+'labels/psd_yolov3_ssd512_adv_8_5_train_70.json',
            pipeline=train_pipeline
        )),
    
ori_train =dict(
        _delete_=True,
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            classes = CLASSES,
            img_prefix=data_root + 'images',
            ann_file=data_root+'labels/psd_yolov3_train_70.json',
            pipeline=train_pipeline
        )),
         

data = dict( 
    train= [ori_train,
    adv_train,],

        
    val=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'),
    test=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'))

runner = dict( max_epochs=12)