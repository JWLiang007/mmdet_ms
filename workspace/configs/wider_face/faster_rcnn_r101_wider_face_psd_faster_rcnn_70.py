_base_ = './faster_rcnn_r50_wider_face_psd_faster_rcnn_70.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
