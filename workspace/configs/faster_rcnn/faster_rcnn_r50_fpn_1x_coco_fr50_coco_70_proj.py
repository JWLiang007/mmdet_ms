_base_ = './faster_rcnn_r50_fpn_1x_proj.py'


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

load_from = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'