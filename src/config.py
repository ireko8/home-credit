"""
Home Credit Default Risk
DenoisingAutoEncoder's code

this code defines file path and constant

this code is based on OsciiArt's DAE kernel
https://www.kaggle.com/osciiart/denoising-autoencoder

writer: Ireko8
"""

TRAIN_PATH = 'input/application_train.csv'
TEST_PATH = 'input/application_test.csv'

cat_cols_804 = [
    'f108_NAME_GOODS_CATEGORY', 'f002_WEEKDAY_APPR_PROCESS_START',
    'f109_NAME_GOODS_CATEGORY', 'f108_NAME_TYPE_SUITE',
    'f109_PRODUCT_COMBINATION', 'f002_NAME_FAMILY_STATUS',
    'f002_OCCUPATION_TYPE', 'f108_PRODUCT_COMBINATION', 'f510_CREDIT_TYPE',
    'f002_WALLSMATERIAL_MODE', 'f002_NAME_INCOME_TYPE',
    'f002_NAME_EDUCATION_TYPE', 'f002_ORGANIZATION_TYPE', 'f509_CREDIT_TYPE'
]
seed = 71

ae_hls = 4096
swap_rate = 0.15
batch_size = 128
num_epoch = 1000
