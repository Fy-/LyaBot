# -*- coding: utf-8 -*-
'''
	LyaBot, Training
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

'''
import os
import math
import tensorflow as tf
from time import time

from model import Model
from model_utils import create_or_load_model, create_train_model, create_infer_model
from misc_utils import format_spm_text, load_data, get_sentence, run_infer_sample
from train_utils import init_stats, update_stats, process_stats, print_step_info, add_summary, calc_num_steps, blue_score
from file_utils import lines_in_file

from settings import settings



def add_summary(summary_writer, global_step, tag, value):
	''' Add a new summary to the current summary_writer. '''
	summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
	summary_writer.add_summary(summary, global_step)

def calc_num_steps(e):
	csize = 0
	for file in ['train_1.bpe.src', 'data.bpe.src']:
		csize += lines_in_file(os.path.join(settings.data_formated, file))

	steps = math.ceil((e+1) * csize / (settings.batch_size))
	return steps

if __name__ == "__main__":

	infer_model = create_infer_model(Model)

	sample_src_data = load_data(os.path.join(settings.data_formated, 'dev.bpe.src'))
	sample_tgt_data = load_data(os.path.join(settings.data_formated, 'dev.bpe.tgt'))


	config = tf.ConfigProto(device_count = {'GPU': 1})
	config.gpu_options.allow_growth = True
	config.intra_op_parallelism_threads = 10
	config.inter_op_parallelism_threads = 10
	
	infer_sess = tf.Session(config=config, graph=infer_model.graph, )


	#run_infer_sample(infer_model, infer_sess, sample_src_data, sample_tgt_data, 5)

	for epoch, learning_rate in enumerate([0.001, 0.0001, 0.00001]):
		settings.learning_rate = learning_rate
		train_model = create_train_model(Model, 'train_1')
		train_sess = tf.Session(config=config, graph=train_model.graph, )

		with train_model.graph.as_default():
			loaded_train_model, global_step = create_or_load_model(train_model.model, train_sess, "train")

		summary_writer = tf.summary.FileWriter(os.path.join(settings.path_model, 'log_dir/'), train_model.graph)

		if not global_step:
			global_step = 0

		last_stats_step = global_step
		last_save_step = global_step
		start_train_time = time()
		stats = init_stats()
		skip_count = settings.batch_size * settings.epoch_step
		train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: skip_count})
		info = {
				"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
				"avg_grad_norm": 0.0,
				"learning_rate": loaded_train_model.learning_rate.eval(session=train_sess)
			}

		if epoch < settings.epoch:
			continue

		num_steps = calc_num_steps(settings.epoch)
		print ('*** Starting at epoch {} for {}/{} steps'.format(settings.epoch, global_step, num_steps))

		while settings.epoch_step < num_steps:
			start_time = time()

			try:
				step_result = loaded_train_model.train(train_sess)
				settings.epoch_step += 1
			except tf.errors.OutOfRangeError:
				print('*** Finished an epoch at step {}'.format(global_step))

				break
			
			global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result)
			summary_writer.add_summary(step_summary, global_step)

			if global_step - last_stats_step >= settings.step_per_show:
				last_stats_step = global_step
				is_overflow = process_stats(stats, info, global_step, settings.step_per_show)
				print_step_info(global_step, info)

				if is_overflow:
					break
				stats = init_stats()


			if global_step - last_save_step >= settings.step_per_save:
				run_infer_sample(infer_model, infer_sess, sample_src_data, sample_tgt_data, 5)

				with open(os.path.join(settings.path_model, 'epoch_step'), 'w') as e:
					e.write(str(settings.epoch_step))


				last_save_step = global_step
				loaded_train_model.saver.save(train_sess, os.path.join(settings.path_model, "model.ckpt"), global_step=global_step)
				add_summary(summary_writer, global_step, "train_ppl", info["train_ppl"])


		settings.epoch += 1
		settings.epoch_step = 0
		train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: 0})
		with open(os.path.join(settings.path_model, 'epoch'), 'w') as e:
			e.write(str(settings.epoch))

		summary_writer.close()
		train_sess.close()

