#!/bin/sh

python train.py -e 100 -lrs 5
python train.py -e 100 -lrs 5 -b 8
python train.py -e 100 -lrs 5 -d 1
python train.py -e 100 -lrs 5 -d 1 -b8
python train.py -e 100 -lrs 5 -d 2
python train.py -e 100 -lrs 5 -d 2 -b8
python train.py -e 100 -lrs 5 -d 3
python train.py -e 100 -lrs 5 -d 3 -b8
python train.py -e 100 -lrs 5 -d 4
python train.py -e 100 -lrs 5 -d 4 -b8
python train.py -e 100 -lrs 7
python train.py -e 100 -lrs 7 -b 8
python train.py -e 100 -lrs 7 -d 1
python train.py -e 100 -lrs 7 -d 1 -b8
python train.py -e 100 -lrs 7 -d 2
python train.py -e 100 -lrs 7 -d 2 -b8
python train.py -e 100 -lrs 7 -d 3
python train.py -e 100 -lrs 7 -d 3 -b8
python train.py -e 100 -lrs 7 -d 4
python train.py -e 100 -lrs 7 -d 4 -b8
