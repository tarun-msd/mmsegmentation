_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
     '../_base_/default_runtime.py',
    '../_base_/datasets/meli.py',
    '../_base_/schedules/schedule_40k.py'
]

checkpoint_config = dict(interval=10, max_keep_ckpts=5)
# model = dict(
#     decode_head=dict(num_classes=2), auxiliary_head=dict(num_classes=2))
#log_config = dict(  # config to register logger hook
#            interval=1, hooks=[dict(type='TensorboardLoggerHook')])
