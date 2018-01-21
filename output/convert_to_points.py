from os import listdir
from os.path import isfile, join
import pandas as pd

logfiles = [f for f in listdir('./') if f.endswith('.log')]
print(logfiles)

outfilename = 'plot/all.csv'
points_matrix = []
columns = []

for idx, logfile in enumerate(logfiles):
    fin = open(logfile, 'r')
    prefix = logfile.split('.')[0]
    columns.append(prefix+'train_loss')
    columns.append(prefix+'val_loss')
    points_matrix.append([])
    points_matrix.append([])

    for line in fin:
        if line.startswith('Epoch:'):
            train_loss_term = line.split('\t')[3]
            train_loss = train_loss_term.split(' ')[2][1:-1]
        elif line.startswith('* Loss'):
            val_loss_term = line.split('\t')[0]
            val_loss = val_loss_term.split(' ')[3][1:-1]
            points_matrix[idx*2].append(train_loss)
            points_matrix[idx*2+1].append(val_loss)

points_df = pd.DataFrame(points_matrix, columns).transpose()
print(points_df.info())
points_df.to_csv(outfilename, index=False)