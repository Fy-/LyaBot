# -*- coding: utf-8 -*-
'''
	LyaBot, Model utils
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	LyaBot
	Copyright (C) 2018 Florian Gasquez <m@fy.to>

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.

	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	This file is based and inspired by https://github.com/tensorflow/nmt
'''

import tensorflow as tf
from tensorflow.python.ops import lookup_ops
import time, os
from .settings import settings
from .iterator_utils import DataIterator
from .vocab import Vocab


def create_eval_model(model_creator):

	graph = tf.Graph()
	with graph.as_default(), tf.container('eval'):
		src_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
		tgt_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
		src_dataset = tf.data.TextLineDataset(src_file_placeholder)
		tgt_dataset = tf.data.TextLineDataset(tgt_file_placeholder)
		vocab_table, _ = Vocab.create_vocab_tables()


		iterator = DataIterator.get_iterator(vocab_table, src_dataset=src_dataset, tgt_dataset=tgt_dataset)

		model = model_creator(mode=tf.contrib.learn.ModeKeys.EVAL, iterator=iterator, vocab_table=vocab_table)

	return  type('',(object,), {
		'graph' : graph,
		'model' : model,
		'iterator' : iterator,
		'src_file_placeholder' : src_file_placeholder,
		'tgt_file_placeholder' :  tgt_file_placeholder
	})

def create_train_model(model_creator, file):
	graph = tf.Graph()
	with graph.as_default(), tf.container("train"):
		skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

		vocab_table, _ = Vocab.create_vocab_tables()
		iterator = DataIterator.get_iterator(vocab_table, file=file, skip_count=skip_count_placeholder)
		model = model_creator(mode=tf.contrib.learn.ModeKeys.TRAIN, iterator=iterator, vocab_table=vocab_table)

	return  type('',(object,), {
		'graph' : graph,
		'model' : model,
		'iterator' : iterator,
		'skip_count_placeholder' :skip_count_placeholder
	})

def create_infer_model(model_creator):
	graph = tf.Graph()

	with graph.as_default(), tf.container("infer"):

		vocab_table, _ = Vocab.create_vocab_tables()
		reverse_vocab_table = lookup_ops.index_to_string_table_from_file(os.path.join(settings.data_formated, 'vocab.bpe.src'), default_value=settings.unk)

		src_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
		batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)
		src_dataset = tf.data.Dataset.from_tensor_slices(src_placeholder)

		iterator = DataIterator.get_infer_iterator(
				src_dataset,
				vocab_table,
				batch_size=batch_size_placeholder
			)

		model = model_creator(
				mode=tf.contrib.learn.ModeKeys.INFER, 
				iterator=iterator, 
				vocab_table=vocab_table, 
				reverse_vocab_table=reverse_vocab_table
			)

	return type('',(object,), {
		'graph' : graph,
		'model' : model,
		'src_placeholder' : src_placeholder,
		'batch_size_placeholder': batch_size_placeholder,
		'iterator' : iterator
	})

def load_model(model, ckpt, session, name):
	start_time = time.time()
	print(ckpt)
	model.saver.restore(session, ckpt)
	session.run(tf.tables_initializer())
	print ('\n*** Loaded {} model with fresh parameters, time {:.2f}s'.format(name, (time.time()-start_time)))
	return model

def create_or_load_model(model, session, name):
	start_time = time.time()
	latest_ckpt = tf.train.latest_checkpoint(settings.path_model)
	if latest_ckpt:
		model = load_model(model, latest_ckpt, session, name)
	else:
		session.run(tf.global_variables_initializer())
		session.run(tf.tables_initializer())
		print ('\n*** Created {} model with fresh parameters, time {:.2f}s'.format(name, (time.time()-start_time)))

	global_step = model.global_step.eval(session=session)
	return model, global_step


def single_cell(num_units, forget_bias, dropout, mode):
	''' It's A BLOB or a Physarum polycephalum '''
	dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0
	#single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
	single_cell = tf.contrib.rnn.GRUCell(num_units)
	if dropout > 0.0:
		 single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=(1.0-dropout))

	return single_cell 

def create_rnn_cell(num_units, num_layers, forget_bias, dropout, mode):
	''' Not a blob anymore :'( '''
	cell_list = []
	for i in range(num_layers):
		scell = single_cell(num_units, forget_bias, dropout, mode)
		cell_list.append(scell)

	if len(cell_list) == 1:
		return cell_list[0]
	else:
		return tf.contrib.rnn.MultiRNNCell(cell_list)


def gradient_clip(gradients, max_gradient_norm):
	''' Clipping gradients of a model.'''
	clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
	gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm)]
	gradient_norm_summary.append(tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients)))

	return clipped_gradients, gradient_norm_summary, gradient_norm
