# -*- coding: utf-8 -*-
'''
	LyaBot, Training
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
'''
import os
import math
import tensorflow as tf
from time import time

from model import Model
from model_utils import create_or_load_model, create_train_model, create_infer_model, create_eval_model
from train_utils import TrainStats, calc_num_steps, blue_score, add_summary
from eval_utils import run_infer_sample, run_eval, run_full_eval
from file_utils import load_data_readlines
from settings import settings


def train_fn(epoch, lr):
	sample_src_data = load_data_readlines(os.path.join(settings.data_formated, 'dev.bpe.src'))
	sample_tgt_data = load_data_readlines(os.path.join(settings.data_formated, 'dev.bpe.tgt'))

	config = settings.get_config_proto()
	settings.num_train_steps = calc_num_steps(epoch)

	settings.learning_rate = lr
	train_model = create_train_model(Model, 'final')
	infer_model = create_infer_model(Model)
	eval_model = create_eval_model(Model)

	train_sess = tf.Session(config=config, graph=train_model.graph, )
	infer_sess = tf.Session(config=config, graph=infer_model.graph, )
	eval_sess = tf.Session(config=config, graph=eval_model.graph, )

	with train_model.graph.as_default():
		loaded_train_model, global_step = create_or_load_model(train_model.model, train_sess, 'train')

	last_stats_step = global_step
	last_save_step = global_step
	last_eval = global_step
	last_feval = global_step

	skip_count = settings.batch_size * settings.epoch_step
	train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: skip_count})
	start_train_time = time()
	train_stats = TrainStats(train_model)

	print ('\nStarting training {}/{}, epoch={}, learning_rate={} ^^\n'.format(settings.epoch_step, settings.num_train_steps, settings.epoch, settings.learning_rate))
	while True:
		start_time = time()

		try:
			step_result = loaded_train_model.train(train_sess)
			settings.epoch_step += 1
		except tf.errors.OutOfRangeError:
			print('*** Finished an epoch at step {}'.format(global_step))
			settings.epoch_step = 0
			break
		
		global_step = train_stats.update(start_time, step_result)

		if global_step - last_stats_step >= settings.step_per_show:
			last_stats_step = global_step
			is_overflow = train_stats.process(global_step, settings.step_per_show)
			train_stats.print_step_info()

			if is_overflow:
				print ('*** Break du to overflow.')
				break

			train_stats.reset()


		if global_step - last_save_step >= settings.step_per_save:
			run_infer_sample(infer_model, infer_sess, sample_src_data, sample_tgt_data, 5)

			settings.save()

			last_save_step = global_step
			loaded_train_model.saver.save(train_sess, os.path.join(settings.path_model, 'model.ckpt'), global_step=global_step)
			add_summary(train_stats.summary_writer, global_step, 'train_ppl', train_stats.info_train_ppl)

		if global_step - last_eval >= settings.step_per_eval:
			last_eval = global_step
			run_eval(eval_model, eval_sess, train_stats.summary_writer)

		if global_step - last_feval >= settings.step_per_feval:
			last_feval = global_step
			run_full_eval(infer_model, infer_sess, train_stats.summary_writer)

	settings.epoch += 1
	settings.epoch_step = 0
	settings.save()
	train_sess.close()
	infer_sess.close()

if __name__ == '__main__':
	for epoch, learning_rate in enumerate(settings.learning_rate):


		train_fn(epoch, learning_rate)
		