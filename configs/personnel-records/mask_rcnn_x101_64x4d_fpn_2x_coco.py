_base_ = '../mask_rcnn/mask_rcnn_x101_32x4d_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnext101_64x4d')),
    roi_head=dict(
         bbox_head=dict(num_classes=1),
         mask_head=dict(num_classes=1))) #change the num_classes in head to match the dataset's annotation


# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('character',)
data = dict(
    train=dict(
        img_prefix='data/generated/',
        classes=classes,
        ann_file='data/input/train90.json'),
    val=dict(
        img_prefix='data/generated/',
        classes=classes,
        ann_file='data/input/train90_val10.json'),
    test=dict(
        img_prefix='data/generated/',
        classes=classes,
        ann_file='data/input/test10.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/swin_large_patch4_window12_384_22k.pth'

