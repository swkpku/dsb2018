import sys
sys.path.append("..")
from dataset.dataset import DSB2018Dataset

TRAIN_DATA_ROOT = '/home/dsb2018/stage1_train_data/'
TEST_DATA_ROOT = '/home/dsb2018/stage1_test_data/'

train_dataset = DSB2018Dataset(TRAIN_DATA_ROOT+'train_ids_train_256_0.txt', 
                            TRAIN_DATA_ROOT+'X_train_256_0.npy',
                            TRAIN_DATA_ROOT+'Y_train_256_0.npy')

id, img, mask = train_dataset[10]