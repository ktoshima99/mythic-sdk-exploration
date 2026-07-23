# Non-destructive patch: makes synthetic CARLA scene names visible to
# nuscenes-devkit's hardcoded train/val split lists (nuscenes.utils.splits).
# Remove this file (or unset PYTHONPATH pointing at its directory) to fully
# revert -- no other files are modified by this patch.
try:
    from nuscenes.utils import splits as _nusc_splits

    _CARLA_TRAIN = ['scene-9001', 'scene-9002', 'scene-9004']
    _CARLA_VAL = ['scene-9003']

    for _name in _CARLA_TRAIN:
        if _name not in _nusc_splits.train:
            _nusc_splits.train.append(_name)
    for _name in _CARLA_VAL:
        if _name not in _nusc_splits.val:
            _nusc_splits.val.append(_name)
except ImportError:
    pass
