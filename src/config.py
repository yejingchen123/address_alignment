from pathlib import Path

#根目录
ROOT_DIR=Path(__file__).parent.parent
# print(ROOT_DIR)

#数据目录
RAW_DATA=ROOT_DIR / 'data' / 'raw'
PROCESSED_DATA=ROOT_DIR / 'data' / 'processed'

#模型目录
CHECKPOINT_BEST_DIR=ROOT_DIR / 'checkpoint' / 'best'
CHECKPOINT_LAST_DIR=ROOT_DIR / 'checkpoint' / 'last'

LOGS_DIR=ROOT_DIR / 'logs'

#模型
PRETRAIN_MODEL_PATH=ROOT_DIR / 'pretrained' / 'roberta-small-wwm-chinese-cluecorpussmall'

# BATCH_SIZE=64
# LEARNING_RATE=1e-5
# EPOCH=2
# SAVE_STEPS=50
# EARLY_STOP_TYPE='acc' #'loss' or 'f1' or 'acc'
# EARLY_STOP_PATIENCE=5
# #自动混合精度
# USE_AMP=False