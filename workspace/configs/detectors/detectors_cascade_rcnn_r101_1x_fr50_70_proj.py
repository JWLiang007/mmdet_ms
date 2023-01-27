_base_ = './detectors_cascade_rcnn_r50_1x_coco.py'




model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        rfp_backbone=dict(
            depth=101,
            pretrained='torchvision://resnet101')),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=14,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),

)
CLASSES = ('person','car','bus','train','truck','camouflage man', 'cruiser/destroyer', 'littoral combat ship','landing helicopte ship','aircraft carrier','fighter', 'patrol aircraft','uav', 'bomber' )
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/psd_fr50_train_70.json',
        # separate_eval=False,
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root+'labels/psd_fr50_test_70.json',
        img_prefix=data_root + 'images',

    ),
    test=dict(
        classes = CLASSES,
        ann_file=data_root+'labels/psd_fr50_test_70.json',
        img_prefix=data_root + 'images',

    )
)

auto_scale_lr = dict(enable=True, base_batch_size=16)