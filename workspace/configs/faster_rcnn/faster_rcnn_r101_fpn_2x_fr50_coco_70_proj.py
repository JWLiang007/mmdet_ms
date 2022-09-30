_base_ = './faster_rcnn_r50_fpn_2x_fr50_70_proj.py'


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


data_root = 'dataset/project/'

data = dict(

    train=dict(
    
        ann_file=data_root+'labels/psd_fr50_coco_train_70.json',

    ),
    val=dict(

        ann_file=data_root+'labels/psd_fr50_coco_test_70.json',


    ),
    test=dict(
 
        ann_file=data_root+'labels/psd_fr50_coco_test_70.json',


    )
)
