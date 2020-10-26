CSV_COLUMNS = [
    'id', 'td',
    'yaw', 'pitch', 'roll',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'qx', 'qy', 'qz', 'qw'
]

SUBJECTS = [
    'albert',
    'canon_12_5',
    'chen',
    'daniel',
    'isa_12_5',
    'joanne',
    'jq_12_6',
    'kelly_11_7',
    'kevin_11_7',
    'ruocheng',
    'russell_11_20_stand',
    'russell_random_12_7',
    'solomon',
    'yiheng_11_30',
    'yiheng_12_5',
    'yongxu_11_30',
    'zhaoye',
    'wenzhou_12_5',
    'haobin_11_22',
    'janet',
    'russell_11_7',
]

LABELS = 'abcdefghijklmnopqrstuvwxyz'
CALI_NAME = 'calibration'

NUM_OF_INTERP_POINTS = 100
THETA_RANGE = 5

SPLIT_MODE_CLASSIC = 0
SPLIT_MODE_BY_SUBJECT = 1