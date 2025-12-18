import sys
sys.path.append('/data1/wangjingtao/workplace/python/pycharm_remote/meta-learning-classfication')


import os
import argparse
import time
import csv

from common.dataloader import *
import common.utils as utils

import dl.dl_func as dlf
import dl.predict_for_lymph as dlf_pre_lymph
import dl.predict_for_oct as dlf_pre_oct


def dl_enter(args, test_data):

    dlf.trainer(args,test_data[0], test_data[1], test_data[2], 0)

def predict_enter(args, test_data, meta_epoch_for_predict):

    if 'lymph' in args.datatype or 'thyroid' in args.datatype or 'ffpe' in args.datatype or \
        'bf' in args.datatype or 'multi_centers' in args.datatype or 'camely' in args.datatype \
            or 'crc' in args.datatype or 'background' in args.datatype:
        if args.isheatmap:
            dlf_pre_lymph.heat_map_fuse_2025(args, meta_epoch_for_predict)
        else:
            dlf_pre_lymph.predict(args, test_data[2], meta_epoch_for_predict)
    elif args.datatype == "oct":
        dlf_pre_oct.predict(args, test_data[2], meta_epoch_for_predict)


