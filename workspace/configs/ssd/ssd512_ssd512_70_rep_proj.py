_base_ = 'ssd512_proj.py'

data_root = 'dataset/project/'
data = dict(
    train=dict(
        times= 4,
        dataset=dict(
            ann_file=data_root+'labels/psd_ssd512_train_70.json',
            )),
    val=dict(        

        ann_file=data_root+'labels/psd_ssd512_test_70.json'),
    test=dict(        

        ann_file=data_root+'labels/psd_ssd512_test_70.json'))
