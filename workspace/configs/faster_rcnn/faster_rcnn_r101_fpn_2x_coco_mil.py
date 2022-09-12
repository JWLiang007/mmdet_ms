_base_ = './faster_rcnn_r101_fpn_2x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=14
        )),

)
CLASSES = ('person bev', 'car bev', 'van bev', 'truck bev','bus bev','person', 'car','aeroplane','bus', 'train' , 'truck',  'boat', 'bird',   'camouflage man')
dataset_type = 'CocoDataset'
data_root = 'dataset/military_new/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/train.json',
        # separate_eval=False,
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root+'labels/val.json',
        img_prefix=data_root + 'images',

    ),
    test=dict(
        classes = CLASSES,
        ann_file=data_root+'labels/val.json',
        img_prefix=data_root + 'images',

    )
)

auto_scale_lr = dict(enable=True, base_batch_size=16)