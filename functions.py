# -*- coding: utf-8 -*-

# import matplotlib.pyplot as plt
import time
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas.core.frame import DataFrame
from Model import get_model, get_callbacks
from data import *
from keras import backend as K
from data import *
from sklearn import metrics
import seaborn as sns
import pandas as pd

init_ratio = 0.4 # gpu volatile ratio supporting parallel running

def cal_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph, run_meta=run_meta, cmd='op', options=opts)
    print('flops'+ str(flops.total_float_ops))

def resolve_dict(hp):
    return hp['pc'], hp['w'], hp['decay'], hp['bs'], hp['lr'], hp['epochs'], hp['disjoint'], hp['model_type']


def record_results(dataID, hp, results):
    pass
    # record_file_name = './' + data_name_dict[str(dataID)] + '_' + str(hp['w']) + '_'  + '.txt'
    # with open(record_file_name, 'w') as f:
    #     f.write('mean_oa:' + str(results[0]) + '\n')
    #     f.write('std_oa:' + str(results[1]))

def CalAccuracy(predict, label):
    n = label.shape[0]
    OA = np.sum(predict == label) * 1.0 / n

    correct_sum = np.zeros((max(label) + 1))
    reali = np.zeros((max(label) + 1))
    predicti = np.zeros((max(label) + 1))
    producerA = np.zeros((max(label) + 1))

    for i in range(0, max(label) + 1):
        correct_sum[i] = np.sum(label[np.where(predict == i)] == i)
        reali[i] = np.sum(label == i)
        predicti[i] = np.sum(predict == i)
        if reali[i] == 0:
            print('Warnning!',str(i),'class have no samples')
            producerA[i] = 1.0
        else:
            producerA[i] = correct_sum[i] / reali[i]

    Kappa = (n * np.sum(correct_sum) - np.sum(reali * predicti)) * 1.0 / (n * n - np.sum(reali * predicti))

    return OA, Kappa, producerA


def loop_train_test(dataID=1, num_list=[], verbose=1, run_times=50, hyper_parameters={}, output_map=False, only_draw_label=False, model_save=False):

    global recall_s
    t = time.time()
    gpu_setting(init_ratio=init_ratio)

    num_PC, w, decay, batch_size, lr, epochs, disjoint, model_type = resolve_dict(hyper_parameters)
    n_class = get_class_num(dataID=dataID)
    results = np.zeros((n_class + 2, run_times))
    run_time = np.zeros((3, run_times))

    for run_i in range(run_times):
        K.clear_session()
        model = get_model(w, w, num_PC=num_PC, nb_classes=n_class, dataID=dataID, type=model_type, lr=lr)
        model.summary()
        callbacks = get_callbacks(decay=decay)

        if disjoint:
            train_batch, train_step, test_batch, test_step = generate_fixed_train_test_batch(dataID=dataID,
                                                                                        num_PC=num_PC,
                                                                                        w=w,
                                                                                        batch_size=batch_size)
        else:
            train_batch, train_step, test_batch, test_step = generate_train_test_batch(dataID=dataID,
                                                                                        num_PC=num_PC,
                                                                                        num_list=num_list,
                                                                                        w=w,
                                                                                        batch_size=batch_size)
        train_time = time.time()
        data = get_images(test_batch, step=test_step)
        gt =  get_labels(test_batch, test_step, argmax=True)
        model.fit_generator(train_batch, steps_per_epoch=train_step, epochs=epochs, verbose=verbose,
                                callbacks=callbacks)

        truth = get_labels(test_batch, test_step, argmax=True)
        print(test_batch)
        train_time = time.time() - train_time

        if model_save:
            save_model(model)

        test_time = time.time()

        subtrainfeature1 = pd.read_csv(r'D:\Path')
        subtrainLabel1 = pd.read_csv(r'D:\Path')

        subtrain = pd.merge(subtrainLabel1, subtrainfeature1, on='Id')
        from sklearn.utils import shuffle
        # subtrain = shuffle(subtrain)
        labels = subtrain.Class
        subtrain.drop(["Class", "Id"], axis=1, inplace=True)
        subtrain = subtrain.values
        (x_train, x_test, y_train, y_test) = train_test_split(subtrain, labels, test_size=0.95, stratify=labels)
        print(y_train)
        new_path3 = r'D:\Path'
        # f = open(new_path3,'w')
        # csv_write = f.writer(y_train,dialect='excel')
        y_train.to_csv(new_path3, index=False, header=False)

        # x_train = subtrain[0:10000]
        x_test = subtrain[0:]
        print(x_test.shape)
        # # y_train = labels[0:10000]
        y_test = labels[0:]

        # x_train = subtrain[0:10000]
        # x_test = subtrain[0:]
        # # y_train = labels[0:10000]
        # y_test = labels[0:]

        # y_train = keras.utils.to_categorical(y_train, num_classes=2)
        # y_test = keras.utils.to_categorical(y_test, num_classes=2)
        print(len(x_train), 'train sequences')
        print(len(x_test), 'test sequences')

        y_train = np_utils.to_categorical(y_train, num_classes=16)
        y_test = np_utils.to_categorical(y_test, num_classes=16)

        prediction = model.predict_generator(test_batch, steps=test_step, verbose=verbose)



        prediction_tsne = model.predict(data)


        data1 = DataFrame(prediction_tsne)
        data1.to_csv(r'D:\Path', index=False, header=False)

        data2 = DataFrame(gt) #输入的test set的标签
        data2.to_csv(r'D:\Path', index=False, header=False)

        prediction = prediction.argmax(axis=-1)
        test_time = time.time() - test_time

        OA, Kappa, ProducerA = CalAccuracy(prediction, truth)

        from sklearn import metrics
        list3 = []

        lines = prediction_tsne.tolist()

        a = 0
        for line in lines:
            if line:
                a = a + 1
                # print(line.index(max(line)))
                list3.append(line.index(max(line)))
                # list.append('\n')

        new_path2 = r'D:\Path'

        list22 = []
        f2 = open(new_path2, mode='r')
        list_true = f2.readlines()

        for i in list_true:
            ii = i
            ii = ii.replace('\n', '')
            ii = int(ii)
            list22.append(ii)
        recall_s = metrics.recall_score(list22, list3, average='macro')


        confusion_matrix = metrics.confusion_matrix(truth, prediction)

        cm = np.array(confusion_matrix)
        matrix_label = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16']
        plt.figure(figsize=(12, 10), dpi=100)
        sns.heatmap(cm, xticklabels=matrix_label, yticklabels=matrix_label, center=0, annot=True, fmt="d",
                    linewidths=.5, cbar=False)

        # plt.title('Indian Pines (15%)', fontsize=22)
        plt.xlabel("1%", fontsize=20)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig('confusion_matrix.png')
        plt.show()

        results[0:n_class, run_i] = ProducerA
        results[-2, run_i] = OA
        results[-1, run_i] = Kappa
        run_time[0, run_i] = train_time
        run_time[1, run_i] = test_time

        print('rand', run_i  + 1, ' ' + data_name_dict[str(dataID)] + ' accuracy:', OA * 100)

        whole_batch, steps = generate_whole_batch(dataID=dataID, num_PC=num_PC, w=w, batch_size=batch_size)
        prediction_time = time.time()
        prediction = model.predict_generator(whole_batch, steps=steps, verbose=verbose)

        prediction_time = time.time() - prediction_time
        run_time[2, run_i] = prediction_time

        Map = prediction.argmax(axis=-1)
        Map1 = elimate_unlabeled_pixel(Map, dataID=dataID, fixed=False)

        row = image_size_dict[str(dataID)][0]
        col = image_size_dict[str(dataID)][1]
        sio.savemat('./' + data_name_dict[str(dataID)] + '_predict', {'y': np.reshape(Map1, (row, col))})
        X_result1 = draw_result(Map1, np.max(prediction, axis=-1), dataID=dataID)
        plt.imsave('./' + data_name_dict[str(dataID)] + '_OA_' + repr(int(OA * 10000)) + '_label.png', X_result1)


        if output_map:
            Map = prediction.argmax(axis=-1)


            row = image_size_dict[str(dataID)][0]
            col = image_size_dict[str(dataID)][1]
            sio.savemat('./' + data_name_dict[str(dataID)]+'_predict',{'y':np.reshape(Map, (row, col))})
            X_result = draw_result(Map, np.max(prediction, axis=-1), dataID=dataID)
            plt.imsave('./' + data_name_dict[str(dataID)] + '_OA_' + repr(int(OA * 10000)) + '.png', X_result)

    mean_oa = np.mean(results[-2] * 100)
    std_oa = np.std(results[-2] * 100)

    aa = np.mean(results[0:n_class], axis=0)
    mean_aa = np.mean(aa) * 100
    std_aa = np.std(aa) * 100

    mean_kappa = np.mean(results[-1])
    std_kappa = np.std(results[-1])

    for i in range(n_class):
        print('Class ', str(i+1),' mean:', str(np.mean(results[i]) * 100), 'std:', str(np.std(results[i]) * 100))

    print('OA mean:', str(mean_oa), 'std:', str(std_oa))
    print('AA mean:', str(mean_aa), 'std:', str(std_aa))
    print('Kappa mean:', str(mean_kappa * 100), 'std:', str(std_kappa * 100))
    print('train_time:', str(np.mean(run_time[0])), 'std:', str(np.std(run_time[0])))
    print('test_time:', str(np.mean(run_time[1])), 'std:', str(np.std(run_time[1])))
    print('prediction_time:', str(np.mean(run_time[2])), 'std:', str(np.std(run_time[2])))
    print('total_time:', str((time.time() - t) / run_times))
    print('recall:', recall_s)
    print('\n')
    # record_results(dataID=dataID, hp=hyper_parameters, results=[mean_oa, std_oa])

def save_model(model, filepath='./kernel_cnn.hdf5'):
    # use model.save_weights can decrease storage
    # but model.save can retraining
    model.save(filepath)


# default 40% GPU
def gpu_setting(init_ratio=0.8):
    if init_ratio > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()

        # auto-increase
        config.gpu_options.per_process_gpu_memory_fraction = init_ratio
        config.gpu_options.allow_growth = True
        # SET Session
        KTF.set_session(tf.Session(config=config))
    else:
        # exchange to cpu mode
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"