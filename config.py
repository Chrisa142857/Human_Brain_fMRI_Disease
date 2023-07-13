BASELINE_MODEL = {
    # 'nclass': 2,
    # 'in_dim': 191, # oasis a2009s
    'nclass': 5,
    'in_dim': 1001, # adni aal_90 (116 for all BOLD)
    'embed_dim': 2048, # 2048 for MLP
    'nhead': 8,
    'nlayer': 16,
}
## Data setup
DATA = {
    'roi_start': 0,
    'roi_end': 90,
    'data_csvn': 'ADNI_AAL90_5class.csv' ,
    # 'roi_start': 41,
    # 'roi_end': 191,
    # 'data_csvn': 'OASIS3_convert_vs_nonconvert.csv',
}
STEP_SIZE = 50
WIN_SIZE = 80
# STEP_SIZE = 100
# WIN_SIZE = 500
## Train-val set
TRAIN_RATIO = 0.7
BATCH_SIZE = 8
DEVICE = 'cuda:1'
SAVE_DIR = './work_dir/ADNI'
learning_rate = 0.000001
num_epochs = 100
