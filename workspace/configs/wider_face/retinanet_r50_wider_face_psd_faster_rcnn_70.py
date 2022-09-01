_base_ = [
    './retinanet_r50_wider_face.py'
]

data_root = 'dataset/WIDERFace/'
data = dict(
    train=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_train_70.json',
    ),
    val=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
    ),
    test=dict(
        ann_file=data_root + 'psd_faster_rcnn_r50_test_70.json',
    )
)