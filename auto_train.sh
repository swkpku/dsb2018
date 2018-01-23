#!/bin/sh

python train.py -e 50 -lrs 6
python train.py -e 50 -lrs 7
python train.py -e 50 -lrs 6 -b 8
python train.py -e 50 -lrs 7 -b 8
