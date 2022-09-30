_base_ = './faster_rcnn_r50_fpn_2x_fr50_70_proj.py'


model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))


