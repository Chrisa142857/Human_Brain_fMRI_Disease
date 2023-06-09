BASELINE_MODEL = {
    'in_dim': 191,
    'embed_dim': 768, # 2048 for MLP
    'nhead': 8,
    'nlayer': 8,
    'nclass': 2
}


## Train-val set

TRAIN_RATIO = 0.7
BATCH_SIZE = 32
DEVICE = 'cuda:0'
SAVE_DIR = './work_dir'
learning_rate = 0.00001
num_epochs = 100