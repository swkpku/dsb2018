from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

data_folder = '/home/iceberg/'

# read json
full_train_df = pd.read_json(data_folder+'train.json')
test_df = pd.read_json(data_folder+'test.json')

# reshape to 75x75
test_df['band_1'] = test_df['band_1'].apply(lambda x: np.array(x).reshape(75, 75))
test_df['band_2'] = test_df['band_2'].apply(lambda x: np.array(x).reshape(75, 75))

# convert inc_angle to number
test_df['inc_angle'] = pd.to_numeric(test_df['inc_angle'], errors='coerce')

print(test_df.info())

test_df.to_json(data_folder+'test_1.json')
