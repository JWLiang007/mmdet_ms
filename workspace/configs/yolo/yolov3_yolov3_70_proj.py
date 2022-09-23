_base_ = './yolov3_proj.py'

data_root = 'dataset/project/'
data = dict(
    train=dict(

        ann_file=data_root+'labels/psd_yolov3_train_70.json',
            ),
    val=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'),
    test=dict(        

        ann_file=data_root+'labels/psd_yolov3_test_70.json'))