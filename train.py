#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os

import tensorflow as tf

import utils
from model import Model
from utils import read_data

from flags import parse_args
FLAGS, unparsed = parse_args()


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s', level=logging.DEBUG)


vocabulary = read_data(FLAGS.text)
print('Data size', len(vocabulary))

# 将字符型vocabulary列表编码为整数vocabulary_int列表
vocabulary_int = utils.vocabulary_to_inter(vocabulary)

with open(FLAGS.dictionary, encoding='utf-8') as inf:
    dictionary = json.load(inf, encoding='utf-8')

with open(FLAGS.reverse_dictionary, encoding='utf-8') as inf:
    reverse_dictionary = json.load(inf, encoding='utf-8')

model = Model(learning_rate=FLAGS.learning_rate, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps)
model.build()

with tf.Session() as sess:
    summary_string_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    saver = tf.train.Saver(max_to_keep=5)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    logging.debug('Initialized')

    try:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.output_dir)
        saver.restore(sess, checkpoint_path)
        logging.debug('restore from [{0}]'.format(checkpoint_path))

    except Exception:
        logging.debug('no check point found....')


    max_steps = ( len(vocabulary_int)//(FLAGS.batch_size*FLAGS.num_steps) )  # 以num_steps为基本单位计算
    step=0
    for epoc in range(1):
        #logging.debug('epoch [{0}]....'.format(epoc))
        state = sess.run(model.state_tensor)   # RNN的起始状态
        for x, y in utils.get_train_data(vocabulary_int, batch_size=FLAGS.batch_size, num_steps=FLAGS.num_steps):

            ##################
            # Your Code here
            ##################
            step += 1
            feed_dict = {model.X: x,
                    model.Y: y,
                    model.keep_prob: 0.85,
                    model.state_tensor: state}

            gs, _, state, l, summary_string = sess.run(
                [model.global_step, model.optimizer, model.outputs_state_tensor, model.loss, model.merged_summary_op], feed_dict=feed_dict)
            summary_string_writer.add_summary(summary_string, gs)

            if gs % (max_steps //10) == 0:
                logging.debug('step [{0}] loss [{1}]'.format(gs, l))

            if gs % (max_steps //4)== 0:
                save_path = saver.save(sess, os.path.join(FLAGS.output_dir, "model.ckpt"), global_step=gs)

            if step>=max_steps:
                break

    summary_string_writer.close()
