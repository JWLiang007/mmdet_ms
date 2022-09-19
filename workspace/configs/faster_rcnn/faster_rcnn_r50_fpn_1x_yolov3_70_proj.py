_base_ = './faster_rcnn_r50_fpn_1x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=18
        )),

)
CLASSES = ('CG','CVN','DDG','LCS','LHA', 'LHD','B52', 'F16', 'F22','F35', 'P8A','RQ4' , 'person','car','bus','train', 'truck','camouflage man')
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,

    train=dict(
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/psd_yolov3_train_70.json',
        # separate_eval=False,
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root+'labels/psd_yolov3_test_70.json',
        img_prefix=data_root + 'images',

    ),
    test=dict(
        classes = CLASSES,
        ann_file=data_root+'labels/psd_yolov3_test_70.json',
        img_prefix=data_root + 'images',

    )
)

auto_scale_lr = dict(enable=True, base_batch_size=16)