_base_ = 'ssd512_proj.py'

data_root = 'dataset/project/'
data = dict(
    train=dict(
        dataset=dict(
            ann_file=data_root+'labels/psd_fr50_train_70.json',
            )),
    val=dict(        

        ann_file=data_root+'labels/psd_fr50_test_70.json'),
    test=dict(        

        ann_file=data_root+'labels/psd_fr50_test_70.json'))

load_from = 'checkpoints/ssd512_coco_20210803_022849-0a47a1ca.pth'