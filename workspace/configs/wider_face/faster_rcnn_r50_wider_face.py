_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/wider_face_coco.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1)),
    rpn_head=dict(
        anchor_generator=dict(scales=[2])
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