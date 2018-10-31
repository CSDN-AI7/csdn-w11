#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf


class Model():
    def __init__(self, learning_rate=0.001,lamda=1e-7,batch_size=16, num_steps=32, num_words=5000, dim_embedding=128, rnn_layers=3):
        r"""初始化函数

        Parameters
        ----------
        learning_rate : float
            学习率.
        lamda : float
            L2正则惩罚参数
        batch_size : int
            batch_size.
        num_steps : int
            RNN有多少个time step，也就是输入数据的长度是多少.
        num_words : int
            字典里有多少个字，用作embeding变量的第一个维度的确定和onehot编码.
        dim_embedding : int
            embding中，编码后的字向量的维度
        rnn_size : int
            每个rnn cell中有多少个神经元
        rnn_layers : int
            有多少个RNN层，在这个模型里，一个RNN层就是一个RNN Cell，各个Cell之间通过TensorFlow提供的多层RNNAPI（MultiRNNCell等）组织到一起
            
        """
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_words = num_words
        self.dim_embedding = dim_embedding
        self.rnn_size = dim_embedding
        self.rnn_layers = rnn_layers
        self.learning_rate = learning_rate
        self.lamda = lamda

    def build(self, embedding_file=None):
        # global step
        self.global_step = tf.Variable(
            0, trainable=False, name='self.global_step', dtype=tf.int64)

        self.X = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='input')
        self.Y = tf.placeholder(
            tf.int32, shape=[None, self.num_steps], name='label')

        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.variable_scope('embedding'):
            if embedding_file:
                # if embedding file provided, use it.
                embedding = np.load(embedding_file)
                embed = tf.constant(embedding, name='embedding')
            else:
                # if not, initialize an embedding and train it.
                embed = tf.get_variable('embedding', [self.num_words, self.dim_embedding])
                tf.summary.histogram('embed', embed)

            data = tf.nn.embedding_lookup(embed, self.X)

        with tf.variable_scope('rnn'):
            ##################
            # Your Code here
            ##################

            # 堆叠多个lstm层，每层神经元个数为rnn_size个
            def single_cell(lstm_size, keep_prob):
                cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
                drop_cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
                return drop_cell
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell(self.rnn_size, self.keep_prob) for _ in range(self.rnn_layers)])
            self.state_tensor = cell.zero_state(self.batch_size, tf.float32)
            # 通过dynamic_rnn对cell展开时间维度
            self.rnn_outputs, self.outputs_state_tensor = tf.nn.dynamic_rnn(cell, data, initial_state=self.state_tensor)
            # self.rnn_outputs的维度是（batch，num_steps，rnn_size），concat后在time_step上减少一个维度
            seq_output = tf.concat(self.rnn_outputs, 1)   #这一步不是必须的，删掉也OK
        # flatten it
        seq_output_final = tf.reshape(seq_output, [-1, self.rnn_size])   # reshape展开之后，维度变为（batch * num_steps）行，rnn_size列

        with tf.variable_scope('softmax'):
            ##################
            # Your Code here
            ##################

            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_words], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_words))
            logits = tf.matmul(seq_output_final, softmax_w) + softmax_b    # 最终shape是 （batch * num_steps）行，num_words列

        tf.summary.histogram('logits', logits)
        self.predictions = tf.nn.softmax(logits, name='predictions')

        # 计算logits_loss
        y_one_hot = tf.one_hot(self.Y, self.num_words) 
        y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
        loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y_reshaped)		
        #loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self.Y, [-1])，logits=logits)
        logits_loss = tf.reduce_mean(loss)
        tf.summary.scalar('logits_loss', logits_loss)

        #计算var_loss
        mean, var = tf.nn.moments(logits, -1)
        var_loss = tf.divide(10.0, 1.0+tf.reduce_mean(var))  # 把标准差作为loss添加到最终的loss里面，避免网络每次输出的语句都是机械的重复
        tf.summary.scalar('var_loss', var_loss)  

        # 计算l2正则loss 
        tv = tf.trainable_variables()#得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
        regularization_cost = self.lamda* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) #0.001是lambda超参数
        tf.summary.scalar('regularization_cost', regularization_cost)
        
        # 总loss
        self.loss = logits_loss + var_loss + regularization_cost 
        tf.summary.scalar('total_loss', self.loss)

        # gradient clip
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer = train_op.apply_gradients(zip(grads, tvars), global_step=self.global_step)

        tf.summary.scalar('loss', self.loss)
        self.merged_summary_op = tf.summary.merge_all()
