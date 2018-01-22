import sys
sys.path.append("..")
from dataset.dataset import DSB2018Dataset
from dataset.transforms import *

TRAIN_DATA_ROOT = '/home/swk/dsb2018/stage1_train_data/'
TEST_DATA_ROOT = '/home/swk/dsb2018/stage1_test_data/'

train_dataset = DSB2018Dataset(TRAIN_DATA_ROOT+'train_ids_train_256_0.txt', 
                            TRAIN_DATA_ROOT+'X_train_256_0.npy',
                            TRAIN_DATA_ROOT+'Y_train_256_0.npy',
                            transform=Compose([
                                RandomRotate(10),                                        
                                RandomHorizontallyFlip()]))

id, img, mask = train_dataset[10]
for i in mask.numpy():
    print(i)
print(mask.numpy())
print(train_dataset.__len__())
