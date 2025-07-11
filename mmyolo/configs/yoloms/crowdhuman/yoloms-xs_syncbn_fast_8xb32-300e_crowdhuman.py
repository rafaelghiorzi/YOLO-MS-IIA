_base_ = '../yoloms-xs_syncbn_fast_8xb32-300e_coco.py'

load_from = 'ckpts/yoloms-xs_syncbn_fast_8xb32-300e_coco.pth'

data_root = 'data/CrowdHuman/'
dataset_type = 'YOLOv5CrowdHumanDataset'

num_classes = 1

anchors = [
    [(6, 14), (12, 28), (19, 48)],  # P3/8
    [(29, 79), (46, 124), (142, 54)],  # P4/16
    [(73, 198), (124, 330), (255, 504)]  # P5/32
]

model = dict(bbox_head=dict(head_module=dict(num_classes=num_classes)))

train_dataloader = dict(dataset=dict(type=dataset_type,
                                     data_root=data_root,
                                     ann_file='annotation_train.odgt',
                                     data_prefix=dict(img='Images/')))

val_dataloader = dict(dataset=dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='annotation_val.odgt',
    data_prefix=dict(img='Images/'),
    # CrowdHumanMetric does not support out-of-order output images
    # for the time being. batch_shapes_cfg does not support.
    batch_shapes_cfg=None))
test_dataloader = val_dataloader

val_evaluator = dict(_delete_=True,
                     type='mmdet.CrowdHumanMetric',
                     ann_file=data_root + 'annotation_val.odgt',
                     metric=['AP', 'MR', 'JI'])
test_evaluator = val_evaluator
