_base_ = 'ssd512_coco.py'
input_size = 512
model = dict(
    bbox_head=dict(
        num_classes=14,))
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'dataset/project/'
CLASSES = ('person bev', 'car bev', 'van bev', 'truck bev','bus bev','person', 'car','aeroplane','bus', 'train' , 'truck',  'boat', 'bird',   'camouflage man')

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        _delete_=True,
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            classes = CLASSES,
            img_prefix=data_root + 'images',
            ann_file=data_root+'labels/train_all.json'
            )),
    val=dict(        
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json'),
    test=dict(        
        type=dataset_type,
        classes = CLASSES,
        img_prefix=data_root + 'images',
        ann_file=data_root+'labels/val_all.json'))


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (8 samples per GPU)
auto_scale_lr = dict(enable=True, base_batch_size=64)
