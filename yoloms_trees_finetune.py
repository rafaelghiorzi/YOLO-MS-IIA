# _base_ = 'yoloms_syncbn_fast_8xb32-300e_coco.py'
_base_ = 'mmyolo/configs/yoloms/yoloms_syncbn_fast_8xb32-300e_coco.py'
# Dataset configuration
data_root = 'D:/UnB/IIA/YOLO-MS-IIA/coco_dataset'
class_name = ('tree',)  # Your single class
num_classes = 1
metainfo = dict(classes=class_name, palette=[(0, 255, 0)])

# Training parameters for fine-tuning
max_epochs = 300  # Reduced from 300 for fine-tuning
train_batch_size_per_gpu = 2  # Adjust based on your GPU memory
val_batch_size_per_gpu = 1
train_num_workers = 2
val_num_workers = 1

# Lower learning rate for fine-tuning
base_lr = 0.0001


# Model configuration - Update for single class
model = dict(
    bbox_head=dict(
        head_module=dict(num_classes=num_classes)
    ),
    train_cfg=dict(
        assigner=dict(num_classes=num_classes)
    )
)

# Data configuration
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_train.json',
        data_prefix=dict(img='images/train/'),
        metainfo=metainfo
    )
)

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_val.json',
        data_prefix=dict(img='images/val/'),
        metainfo=metainfo,
        test_mode=True
    )
)

# Add test dataloader
test_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/instances_test.json',
        data_prefix=dict(img='images/test/'),
        metainfo=metainfo,
        test_mode=True
    )
)

# Evaluators
val_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=f'{data_root}/annotations/instances_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

test_evaluator = dict(
    type='mmdet.CocoMetric',
    ann_file=f'{data_root}/annotations/instances_test.json',
    metric='bbox',
    format_only=False,
    classwise=True,
)

# Training config
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=20
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
    
# Auto-scaling learning rate
auto_scale_lr = dict(enable=True, base_batch_size=16)

# Hooks configuration
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=20,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=5
    ),
    logger=dict(type='LoggerHook', interval=20)
)

# Load pretrained weights for fine-tuning
load_from = 'D:/UnB/IIA/YOLO-MS-IIA/pretrained.pth'