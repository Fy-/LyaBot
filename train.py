# -*- coding: utf-8 -*-
"""
	LyaBot, Data Formatter
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

"""
import os
import tensorflow as tf

from model import Model
from model_utils import create_or_load_model, create_train_model, create_infer_model
from misc_utils import safe_exp, format_spm_text, load_data, get_sentence, run_infer_sample
from settings import settings


def init_stats():
	"""Initialize statistics that we want to accumulate."""
	return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0, "total_count": 0.0, "grad_norm": 0.0}

def update_stats(stats, start_time, step_result):
	(_, step_loss, step_predict_count, step_summary, global_step, step_word_count, batch_size, grad_norm, learning_rate) = step_result
	stats["step_time"] += (time() - start_time)
	stats["loss"] += (step_loss * batch_size)
	stats["predict_count"] += step_predict_count
	stats["total_count"] += float(step_word_count)
	stats["grad_norm"] += grad_norm
	return global_step, learning_rate, step_summary

def process_stats(stats, info, global_step, steps_per_stats):
	info["avg_step_time"] = stats["step_time"] / steps_per_stats
	info["avg_grad_norm"] = stats["grad_norm"] / steps_per_stats
	info["train_ppl"] = safe_exp(stats["loss"] / stats["predict_count"])
	info["speed"] = stats["total_count"] / (1000 * stats["step_time"])

	is_overflow = False
	train_ppl = info["train_ppl"]
	if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
		is_overflow = True
	return is_overflow

def print_step_info(global_step, info):
	print('step {}, lr {}, step-time {:2f}s, wps {:2f}fK, ppl {:2f}, gN {:2f}'.format(
				global_step, info["learning_rate"], 
				info["avg_step_time"], info["speed"], 
				info["train_ppl"], info["avg_grad_norm"]
			)
		)

def add_summary(summary_writer, global_step, tag, value):
	"""Add a new summary to the current summary_writer.
	Useful to log things that are not part of the training graph, e.g., tag=BLEU.
	"""
	summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
	summary_writer.add_summary(summary, global_step)

if __name__ == "__main__":
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	train_model = create_train_model(Model, 'train_1')
	infer_model = create_infer_model(Model)

	sample_src_data = load_data(os.path.join(settings.data_formated, 'dev.bpe.src'))
	sample_tgt_data = load_data(os.path.join(settings.data_formated, 'dev.bpe.tgt'))


	config = tf.ConfigProto(allow_soft_placement=True)
	config.gpu_options.allow_growth = True
	config.intra_op_parallelism_threads = 8
	config.inter_op_parallelism_threads = 8
	train_sess = tf.Session(config=config, graph=train_model.graph, )
	infer_sess = tf.Session(config=config, graph=infer_model.graph, )


	with train_model.graph.as_default():
		loaded_train_model, global_step = create_or_load_model(train_model.model, train_sess, "train")

	summary_writer = tf.summary.FileWriter(os.path.join(settings.path_model, 'log_dir/'), train_model.graph)

	if not global_step:
		global_step = 0

	info = {
			"train_ppl": 0.0, "speed": 0.0, "avg_step_time": 0.0,
			"avg_grad_norm": 0.0,
			"learning_rate": loaded_train_model.learning_rate.eval(session=train_sess)
		}


	run_infer_sample(infer_model, infer_sess, sample_src_data, sample_tgt_data, 5)


	last_stats_step = global_step
	last_save_step = global_step
	start_train_time = time()
	stats = init_stats()
	skip_count = settings.hparams.batch_size * settings.hparams.epoch_step
	train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: skip_count})

	while global_step < settings.hparams.num_train_steps:
		start_time = time()

		try:
			step_result = loaded_train_model.train(train_sess)
			settings.hparams.epoch_step += 1
		except tf.errors.OutOfRangeError:
			settings.hparams.epoch_step = 0
			print('*** Finished an epoch at step {}'.format(global_step))
			train_sess.run(train_model.iterator.initializer, feed_dict={train_model.skip_count_placeholder: 0})
			continue
		
		global_step, info["learning_rate"], step_summary = update_stats(stats, start_time, step_result)
		summary_writer.add_summary(step_summary, global_step)

		if global_step - last_stats_step >= settings.step_per_shos:
			last_stats_step = global_step
			is_overflow = process_stats(stats, info, global_step, steps_per_stats)
			print_step_info(global_step, info)

			if is_overflow:
				break
			stats = init_stats()


		if global_step - last_save_step >= settings.step_per_save:
			run_infer_sample(infer_model, infer_sess, sample_src_data, sample_tgt_data, 5)

			with open(os.path.join(settings.path_model, 'epoch'), 'w') as e:
				e.write(settings.hparams.epoch_step)

			last_save_step = global_step
			loaded_train_model.saver.save(train_sess, os.path.join(settings.path_model, "model.ckpt"), global_step=global_step)
			_add_summary(summary_writer, global_step, "train_ppl", info["train_ppl"])

	summary_writer.close()



