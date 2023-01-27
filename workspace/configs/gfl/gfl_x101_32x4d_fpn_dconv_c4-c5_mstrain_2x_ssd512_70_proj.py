_base_ = './gfl_x101_32x4d_fpn_dconv_c4-c5_mstrain_2x_coco.py'


model = dict(
    bbox_head=dict(
        num_classes=14)
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
        ann_file=data_root+'labels/psd_ssd512_train_70.json',
        # separate_eval=False,
    ),
    val=dict(
        classes=CLASSES,
        ann_file=data_root+'labels/psd_ssd512_test_70.json',
        img_prefix=data_root + 'images',

    ),
    test=dict(
        classes = CLASSES,
        ann_file=data_root+'labels/psd_ssd512_test_70.json',
        img_prefix=data_root + 'images',

    )


)

auto_scale_lr = dict(enable=True, base_batch_size=16)