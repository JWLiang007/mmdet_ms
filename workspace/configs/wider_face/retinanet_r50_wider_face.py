_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/wider_face_coco.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=1,
        anchor_generator=dict(
            octave_base_scale=2))
)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

auto_scale_lr = dict(enable=True)