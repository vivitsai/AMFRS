# -*- coding: utf-8 -*-

import os
from functions import loop_train_test
from data import image_size_dict as dims
from data import draw_false_color, draw_gt, draw_bar

# remove abundant output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

## global constants
verbose = 1 # whether or not print redundant info (1 if and only if in debug mode, 0 in run mode)
run_times = 1 # random run times, recommend at least 10
output_map = True
only_draw_label = True # whether or not only predict labeled samples
disjoint = False # whether or not train and test on spatially disjoint samples

lr = 1e-3 # init learing rate
decay = 1e-3 # exponential learning rate decay
ws = 39 # window size
epochs = 128 # epoch
batch_size = 12 # batch size
model_type = 'AMFRS'  # model type in {'AMFRS', 'demo'}

def pavia_university_experiment():
    hp = {
        'pc': dims['1'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [663, 1864, 209, 306, 134, 502, 133, 368, 94]
    loop_train_test(dataID=1, num_list=num_list, verbose=verbose, run_times=run_times,
                    hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def indian_pine_experiment():
    hp = {
        'pc': dims['2'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    num_list = [4, 142, 83, 23, 48, 73, 4, 47, 4, 97, 245, 59, 20, 126, 38, 9]
    loop_train_test(dataID=2, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=True)


def houston_university_experiment():
    hp = {
        'pc': dims['3'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    num_list = [125, 125, 70, 124, 124, 33, 127, 123, 125, 123, 124, 124, 47, 43, 66]
    loop_train_test(dataID=3, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)

def Salinas_experiment():
    hp = {
        'pc': dims['4'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }

    num_list = [19, 35, 17, 12, 25, 37, 33, 80, 60, 30, 9, 17, 6, 7, 70, 16]
    loop_train_test(dataID=4, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)



def WHU_Hi_HongHu_experiment():
    hp = {
        'pc': dims['5'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    num_list = [140, 35, 218, 1632, 62, 445, 241, 40, 108, 123, 110, 89, 225, 73, 10, 72, 30, 32, 87, 34, 13,40]
    loop_train_test(dataID=5, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def QUH_Qingyun_experiment():
    hp = {
        'pc': dims['6'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }
    num_list = [13907, 8975, 689, 488, 10886, 12797]
    loop_train_test(dataID=6, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)


def QUH_Pingan_experiment():
    hp = {
        'pc': dims['7'][2],
        'w': ws,
        'decay': decay,
        'bs': batch_size,
        'lr': lr,
        'epochs': epochs,
        'disjoint': disjoint,
        'model_type': model_type,
    }


    num_list = [50] * 10
    loop_train_test(dataID=7, num_list=num_list, verbose=verbose, run_times=run_times,
                        hyper_parameters=hp, output_map=output_map, only_draw_label=only_draw_label, model_save=False)



# pavia_university_experiment()
indian_pine_experiment()
# houston_university_experiment()
# Salinas_experiment()
# WHU_Hi_HongHu_experiment()
# QUH_Qingyun_experiment()
# QUH_Pingan_experiment()

# draw_false_color(dataID=1)
draw_false_color(dataID=2)
# draw_false_color(dataID=3)
# draw_false_color(dataID=4)
# draw_false_color(dataID=5)
# draw_false_color(dataID=6)
# draw_false_color(dataID=7)
#
# draw_bar(dataID=1)
# draw_bar(dataID=2)
# draw_bar(dataID=3)
# draw_bar(dataID=4)
# draw_bar(dataID=5)
# draw_bar(dataID=6)
# draw_bar(dataID=7)

# draw_gt(dataID=1, fixed=disjoint)
# draw_gt(dataID=2, fixed=disjoint)
# draw_gt(dataID=3, fixed=disjoint)

