dataset_type = 'MeliDataset'
train_data_root = '/media/train_hdd5/tarun/data/meli_iter11/train'
val_data_root = '/media/train_hdd5/tarun/data/meli_iter11/val'
data_root = train_data_root
train_split_file = '/media/train_hdd5/tarun/data/meli_iter11/train.txt'
val_split_file = '/media/train_hdd5/tarun/data/meli_iter11/val.txt'

img_dir = 'images'
ann_dir = 'labels'
9
img_norm_cfg = dict(
    mean=[103.94, 116.78, 123.68], std=[57.38, 57.12, 58.40], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512,512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=train_data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=train_pipeline,
        split = train_split_file),
    val=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split = val_split_file),
    test=dict(
        type=dataset_type,
        data_root=val_data_root,
        img_dir=img_dir,
        ann_dir=ann_dir,
        pipeline=test_pipeline,
        split = val_split_file))