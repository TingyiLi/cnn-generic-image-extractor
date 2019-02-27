# -*- coding:utf-8 -*-
"""
Based on Google inception_resnet_v2.ckpt
Feature extraction from images，image size:299*299*3
Inception model input: BATCH_SIZE * 8 * 8 * 1536
"""

import inception_resnet_v2
from matplotlib import pyplot as plt

import tensorflow as tf
import pandas as pd
import threading
import xlrd
import os
import time
import PIL.Image as Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
INFO = logging.info

slim = tf.contrib.slim
scope = inception_resnet_v2.inception_resnet_v2_arg_scope

global_para = {}


# feature extractor & classifier
class CLASSIFIER_BY_PATIENT_STATE:
    def __init__(self, CHECKPOINT_PATH, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH, FEATURE_OUTPUT_PATH=None, TRAIN=False,
                 VISUALIZE=False, VALIDATION_IMAGE_PATH=None, VALIDATION_LABEL_PATH=None):
        """
        :param CHECKPOINT_PATH:inception model path
        :param TRAIN_IMAGE_PATH:training set
        :param TRAIN_LABEL_PATH:training set label
        :param VALIDATION_LABEL_PATH:verification
        :param FEATURE_OUTPUT_PATH:output path
        :param TRAIN:train or not
        :param VISUALIZE:visualized feature
        :param VALIDATION_IMAGE_PATH:verification path
        """
        self.__TRAIN = TRAIN
        self.__VISUALIZE = VISUALIZE
        self.__FEATURE_OUTPUT_PATH = FEATURE_OUTPUT_PATH
        self.__CHECKPOINT_PATH = CHECKPOINT_PATH
        self.__TRAIN_IMAGE_PATH = TRAIN_IMAGE_PATH
        self.__TRAIN_LABEL_PATH = TRAIN_LABEL_PATH
        self.__VALIDATION_LABEL_PATH = VALIDATION_LABEL_PATH
        self.__VALIDATION_IMAGE_PATH = VALIDATION_IMAGE_PATH
        # inception_v4 default：输入图片的大小和通道
        self.__HEIGHT = 299
        self.__WIDTH = 299
        self.__CHANNELS = 3
        # inception_v4 输出层名称
        if self.__TRAIN:
            self.__OUTPUT_LAYER_NAME = 'PreLogitsFlatten'
        else:
            self.__OUTPUT_LAYER_NAME = 'FEATURE_NET'
        # 训练模型的参数
        self.__BATCH_SIZE = self.__util_get_image_num(self.__TRAIN_IMAGE_PATH)
        self.__LEARNING_RATE_BASE = 0.8
        self.__LEARNING_RATE_DECAY = 0.99
        self.__REGULARIZATION_RATE = 0.0001
        self.__TRAINING_STEPS = 30000
        self.__MOVING_AVERAGE_DECAY = 0.99
        self.__MODEL_SAVE_PATH = 'model'
        self.__MODEL_NAME = 'model'
        self.__REGULARIZER = None
        # 评估模型参数
        self.__EVAL_INTERVAL_SECS = 10
        # 模型节点参数
        self.__INPUT_NODE = 170
        self.__OUTPUT_NODE = 2
        self.__LAYER1_NODE = 500
        # 程序运行参数
        self.__TRAIN_FLAG = False
        self.__DEBUG_SIZE = None
        self.__PATH = ''
        global global_para

    # 获取图片数量
    def __util_get_image_num(self, PATH):
        return len(os.listdir(PATH))

    # 读入图片
    def __util_get_image_array(self, PATH, start, end):
        files = os.listdir(PATH)
        num = len(files)
        if end > num:
            end = num
        ima = []
        for index, file in enumerate(files):
            if index + 1 < start:
                continue
            elif index + 1 <= end:
                path = os.path.join(PATH, file)
                image = Image.open(path)
                ima_L1 = np.array(image.convert('L'))
                ima_L2 = np.array(image.convert('L'))
                ima_L3 = np.array(image.convert('L'))
                A = np.zeros((len(ima_L1), len(ima_L1), 3))
                A[:, :, 0] = ima_L1
                A[:, :, 1] = ima_L2
                A[:, :, 2] = ima_L3
                ima.append(A)
            else:
                break
        return ima

    # 使用预训练模型提取特征
    def __util_run_inception_resnet_v2(self, PATH, SIZE):
        # 构造输入
        X = tf.placeholder(tf.float32, shape=[SIZE, self.__HEIGHT, self.__WIDTH, self.__CHANNELS])
        with slim.arg_scope(scope()):
            logits, end_points = inception_resnet_v2.inception_resnet_v2(X, is_training=False)
        # 载入预训练参数
        INFO('载入预训练模型：%s' % os.path.basename(self.__CHECKPOINT_PATH))
        saver = tf.train.Saver()
        sess = tf.Session()
        saver.restore(sess, self.__CHECKPOINT_PATH)
        # 目标输出层
        predictions = end_points[self.__OUTPUT_LAYER_NAME]
        # 读取输入图片
        INFO('读取输入图片：%s' % PATH)
        xs = self.__util_get_image_array(PATH, 1, SIZE)
        INFO('运行GOOGLE inception_resnet_v2 模型提取特征...')
        output = sess.run(predictions, feed_dict={X: xs})
        return output

    # 获取特征
    def __get_features(self):
        # 调用预训模型获得结果
        # SIZE * 8 * 8 * 1536
        output = self.__util_run_inception_resnet_v2(self.__TRAIN_IMAGE_PATH,self.__util_get_image_num(self.__TRAIN_IMAGE_PATH))
        # 可视化
        if self.__VISUALIZE:
            INFO('启动可视化线程...')
            threading.Thread(target=self.__util_visualizer, args=(output,)).start()
        INFO('保存结果：%s' % self.__FEATURE_OUTPUT_PATH)
        reshape_output = np.reshape(output, (int(output.size / self.__BATCH_SIZE), self.__BATCH_SIZE))
        df = pd.DataFrame(reshape_output)
        df.to_csv(self.__FEATURE_OUTPUT_PATH + '/features.csv', index=False)
        INFO('保存结果完成：%s' % (self.__FEATURE_OUTPUT_PATH + '/features.csv'))

    # 可视化特征
    def __util_visualizer(self, net):
        shape = net.shape
        batch_size = shape[0]
        depth = shape[3]
        reshaped_net = net[0, :, :, 0]
        for j in range(depth - 1):
            reshaped_net = np.hstack((reshaped_net, net[0, :, :, j + 1]))
        for i in range(batch_size - 1):
            tmp_net = net[i + 1, :, :, 0]
            for j in range(depth - 1):
                tmp_net = np.hstack((tmp_net, net[i + 1, :, :, j + 1]))
            reshaped_net = np.vstack((reshaped_net, tmp_net))

        ima = Image.fromarray(reshaped_net).convert('L')
        image = Image.merge("RGB", (ima, ima, ima))
        image.show()

    # 获取权重参数
    def __util_get_weight_variable(self, shape):
        weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.001))
        if self.__REGULARIZER is not None:
            tf.add_to_collection('losses', self.__REGULARIZER(weights))
        return weights

    # 读取特征
    def __util_load_feature_from_excel(self, path, TRAIN=True):
        book = xlrd.open_workbook(path)
        sheet = book.sheet_by_name('Sheet1')
        if TRAIN:
            cols = int(sheet.ncols * 0.7)
        else:
            cols = sheet.ncols - int(sheet.ncols * 0.7)
        values = []
        label = []
        for col in range(cols - 1):
            values.append(sheet.col_values(col, 1, 171))
            label.append(sheet.cell_value(col, 0))
        return values, label

    # 读取特征
    # 获取label
    def __util_get_label(self, path, num):
        book = xlrd.open_workbook(path)
        sheet = book.sheet_by_index(0)
        state = sheet.row_values(0)
        if num > len(state):
            num = len(state)
        return [[int(v), 1 - int(v)] for v in state[0:num]]

    # 获取前向传播结果
    def __inference(self, x):
        # 第一层前向传播
        with tf.variable_scope('layer1'):
            weights = self.__util_get_weight_variable([self.__INPUT_NODE, self.__LAYER1_NODE])
            biases = tf.get_variable("biases", [self.__LAYER1_NODE], initializer=tf.constant_initializer(0.0))
            layer1 = tf.nn.relu6(tf.matmul(x, weights) + biases)
        # 第二层前向传播
        with tf.variable_scope('layer2'):
            weights = self.__util_get_weight_variable([self.__LAYER1_NODE, self.__OUTPUT_NODE])
            biases = tf.get_variable("biases", [self.__OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
            layer2 = tf.matmul(layer1, weights) + biases
        return layer2

    # 训练模型
    def __train(self):
        # 选择的图片数量
        if self.__DEBUG_SIZE is None:
            SIZE = self.__util_get_image_num(self.__TRAIN_IMAGE_PATH)
        else:
            SIZE = self.__DEBUG_SIZE
        # 使用 inception_v4 模型获取特征值
        try:
            output = self.__util_run_inception_resnet_v2(self.__TRAIN_IMAGE_PATH, SIZE)
            INFO('提取特征完成')
        except Exception:
            raise Exception('提取特征出错')
        # label
        try:
            ys = self.__util_get_label(self.__TRAIN_LABEL_PATH, SIZE)
        except Exception:
            raise Exception('提取Label出错，请检查label数量与图片数量是否匹配')
        INFO('开始训练模型...')
        # 构造输入
        x = tf.placeholder(tf.float32, [None, self.__INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, self.__OUTPUT_NODE], name='y-input')
        # 正则化类：l2正则
        self.__REGULARIZER = tf.contrib.layers.l2_regularizer(self.__REGULARIZATION_RATE)
        y = self.__inference(x)
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(self.__MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.clip_by_value(y, 1e-10, 1),
                                                                       labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        learning_rate = tf.train.exponential_decay(
            self.__LEARNING_RATE_BASE,
            global_step,
            self.__util_get_image_num(self.__TRAIN_IMAGE_PATH) / self.__BATCH_SIZE, self.__LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        saver = tf.train.Saver()
        LOSS = []
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(self.__TRAINING_STEPS):
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: output, y_: ys})
                if i % 10 == 0:
                    INFO("训练 %d 轮后, 训练集上的损失为： %g." % (step, loss_value))
                    LOSS.append(loss_value)
                    global_para['loss'] = LOSS
                    saver.save(sess, os.path.join(self.__MODEL_SAVE_PATH, self.__MODEL_NAME), global_step=global_step)
                    self.__TRAIN_FLAG = True

    # 训练特征
    def __train2(self, batch_size=100):
        INFO('读取数据...')
        values, label = self.__util_load_feature_from_excel(self.__PATH, True)
        X = np.reshape(values, (len(values), len(values[0])))
        Y = np.reshape(label, (len(label), 1))
        # YY = []
        # for k in Y:
        #     tmp = [0, 0, 0, 0]
        #     tmp[int(k) - 1] = 1
        #     YY.append(tmp)
        YY = [[int(k), 1 - int(k)] for k in Y]
        Y = np.reshape(YY, (len(YY), 2))
        num_x = len(X)
        # 构造输入
        x = tf.placeholder(tf.float32, [None, self.__INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, self.__OUTPUT_NODE], name='y-input')
        # 正则化类：l2正则
        self.__REGULARIZER = tf.contrib.layers.l2_regularizer(self.__REGULARIZATION_RATE)
        y = self.__inference(x)
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(self.__MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf.clip_by_value(y, 1e-10, 1),
                                                                       labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        if self.__REGULARIZER is not None:
            loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        else:
            loss = cross_entropy_mean
        learning_rate = tf.train.exponential_decay(
            self.__LEARNING_RATE_BASE,
            global_step,
            num_x / batch_size, self.__LEARNING_RATE_DECAY,
            staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')
        saver = tf.train.Saver()
        LOSS = []

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            k = 0
            for i in range(self.__TRAINING_STEPS):
                start = batch_size * k
                end = batch_size * (k + 1)
                if end > num_x:
                    xs = X[start:num_x, :]
                    ys = Y[start:num_x, :]
                    k = 0
                else:
                    xs = X[start:end, :]
                    ys = Y[start:end, :]
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
                if i % 10 == 0:
                    INFO("训练 %d 轮后, 训练集上的损失为： %g." % (step, loss_value))
                    LOSS.append(loss_value)
                    global_para['loss'] = LOSS
                    saver.save(sess, os.path.join(self.__MODEL_SAVE_PATH, self.__MODEL_NAME), global_step=global_step)
                    self.__TRAIN_FLAG = True

    # 评估函数：从预训练
    def __evaluate2(self):
        # 等待训练进程结果
        while not self.__TRAIN_FLAG:
            time.sleep(1)
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [None, self.__INPUT_NODE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, self.__OUTPUT_NODE], name='y-input')
            INFO('读取数据...')
            values, label = self.__util_load_feature_from_excel(self.__PATH, False)
            X = np.reshape(values, (len(values), len(values[0])))
            Y = np.reshape(label, (len(label), 1))
            YY = []
            # for k in Y:
            #     tmp = [0, 0, 0, 0]
            #     tmp[int(k) - 1] = 1
            #     YY.append(tmp)
            YY = [[int(k), 1 - int(k)] for k in Y]
            Y = np.reshape(YY, (len(YY), 2))
            validate_feed = {x: X, y_: Y}
            y = self.__inference(x)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(self.__MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            accu = []
            while True:
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(self.__MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                        INFO("训练 %s 轮后, 验证集正确率： %g" % (global_step, accuracy_score))
                        accu.append(accuracy_score)
                        global_para['accuracy'] = accu
                    else:
                        INFO('找不到特定模型')
                        return
                time.sleep(self.__EVAL_INTERVAL_SECS)

    # 评估模型效果
    def __evaluate(self):
        # 等待训练进程结果
        while not self.__TRAIN_FLAG:
            time.sleep(1)
        if self.__DEBUG_SIZE is None:
            SIZE = self.__util_get_image_num(self.__VALIDATION_IMAGE_PATH)
        else:
            SIZE = self.__DEBUG_SIZE
        try:
            output = self.__util_run_inception_resnet_v2(self.__VALIDATION_IMAGE_PATH, SIZE)
            INFO('提取特征完成')
        except Exception:
            raise Exception('提取特征出错')
            # label
        try:
            ys = self.__util_get_label(self.__VALIDATION_LABEL_PATH, SIZE)
        except Exception:
            raise Exception('提取Label出错，请检查label数量与图片数量是否匹配')
        with tf.Graph().as_default():
            x = tf.placeholder(tf.float32, [None, self.__INPUT_NODE], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, self.__OUTPUT_NODE], name='y-input')
            validate_feed = {x: output, y_: ys}

            y = self.__inference(x)
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            variable_averages = tf.train.ExponentialMovingAverage(self.__MOVING_AVERAGE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            accu = []
            while True:
                with tf.Session() as sess:
                    ckpt = tf.train.get_checkpoint_state(self.__MODEL_SAVE_PATH)
                    if ckpt and ckpt.model_checkpoint_path:
                        saver.restore(sess, ckpt.model_checkpoint_path)
                        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                        accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                        INFO("训练 %s 轮后, 验证集正确率： %g" % (global_step, accuracy_score))
                        accu.append(accuracy_score)
                        global_para['accuracy'] = accu
                    else:
                        INFO('找不到特定模型')
                        return
                time.sleep(self.__EVAL_INTERVAL_SECS)

    # 程序运行接口
    def run(self):
        if self.__TRAIN:
            INFO('启动训练进程')
            threading.Thread(target=self.__train2, args=()).start()
            # 验证线程
            if self.__VALIDATION_IMAGE_PATH is not None:
                INFO('启动验证线程')
                threading.Thread(target=self.__evaluate2, args=()).start()
        else:
            # 获取特征
            self.__get_features()


# 可视化训练过程
def visualize_para(pltt):
    pltt.figure(1)
    # 第一个图
    pltt.subplot(121)
    pltt.xlabel('10 * STEPS')
    pltt.ylabel('loss')
    pltt.title('Figure 1: LOSS--STEPS')
    pltt.plot(global_para['loss'])
    # 第二个图
    pltt.subplot(122)
    pltt.xlabel('10 * STEPS')
    pltt.ylabel('accuracy')
    pltt.title('Figure 2: ACCURACY--STEPS')
    pltt.plot(global_para['accuracy'])
    pltt.show(block=False)
    pltt.pause(10)
    pltt.close()


# 可视化训练过程
def Visualizer():
    visualize_para(plt)


# 程序接口
def main():
    # 预训练模型地址
    CHECKPOINT_PATH = r'./inception_resnet_v2.ckpt'
    # 训练集、验证集数据及label
    TRAIN_IMAGE_PATH = './train'
    VALIDATION_IMAGE_PATH = './validation'
    TRAIN_LABEL_PATH = './label_train.xlsx'
    VALIDATION_LABEL_PATH = './label_validation.xlsx'
    #  特征输出地址
    FEATURE_OUTPUT_PATH = './'
    # 为True时用来训练分类器
    TRAIN = False
    # 为True时用来可视化训练过程
    VISUALIZE = False
    my_classifier = CLASSIFIER_BY_PATIENT_STATE(CHECKPOINT_PATH, TRAIN_IMAGE_PATH, TRAIN_LABEL_PATH,
                                                FEATURE_OUTPUT_PATH, TRAIN,
                                                VISUALIZE, VALIDATION_IMAGE_PATH, VALIDATION_LABEL_PATH)
    # 启动类
    my_classifier.run()


if __name__ == '__main__':
    # 程序信息
    INFO(__doc__)
    # INFO(__author__)
    # 启动主类线程
    threading.Thread(target=main, args=()).start()
    # 循环显示结果
    # 当训练时用来可视化
    TRAIN = False
    while TRAIN:
        try:
            Visualizer()
        except Exception:
            INFO('可视化线程未获得数据！')
            time.sleep(10)
