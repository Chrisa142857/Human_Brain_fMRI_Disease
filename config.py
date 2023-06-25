BASELINE_MODEL = {
    'nclass': 2,
    'in_dim': 191, # oasis a2009s
    # 'nclass': 5,
    # 'in_dim': 90, # adni aal_90 (116 for all BOLD)
    'embed_dim': 768, # 2048 for MLP
    'nhead': 8,
    'nlayer': 16,
}

## Data setup
# STEP_SIZE = 50
# WIN_SIZE = 80
STEP_SIZE = 100
WIN_SIZE = 500
## Train-val set
TRAIN_RATIO = 0.7
BATCH_SIZE = 32
DEVICE = 'cuda:2'
SAVE_DIR = './work_dir'
learning_rate = 0.00001
num_epochs = 100
