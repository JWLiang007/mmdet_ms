_base_ = './faster_rcnn_r50_fpn_1x_proj.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=5.0),
        ))
)
data_root = 'dataset/project/'

data = dict(

    train=dict(
    
        ann_file=data_root+'labels/psd_fr50_train_70.json',

    ),
    val=dict(

        ann_file=data_root+'labels/psd_fr50_test_70.json',


    ),
    test=dict(
 
        ann_file=data_root+'labels/psd_fr50_test_70.json',


    )
)

