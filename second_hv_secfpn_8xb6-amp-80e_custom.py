_base_ = [
    '../_base_/models/second_hv_secfpn_custom.py',
    '../_base_/datasets/custom.py',
    '../_base_/schedules/cyclic-40e.py', '../_base_/default_runtime.py'
]

# schedule settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale=4096.)
