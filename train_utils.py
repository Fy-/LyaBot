import nltk
import math
from time import time

def safe_exp(value):
	''' Exponentiation with catching of overflow error. (And not The Barrens) '''
	try:
		ans = math.exp(value)
	except OverflowError:
		ans = float("inf")
	return ans

def init_stats():
	''' Initialize statistics that we want to accumulate. '''
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
	print('step {}, lr {}, tps {:2f}s, {:2f} wps, ppl {:2f}, gN {:2f}'.format(
				global_step, info["learning_rate"], 
				info["avg_step_time"], info["speed"], 
				info["train_ppl"], info["avg_grad_norm"]
			)
		)

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

def blue_score(labels, predictions):

	#nltk.translate.bleu_score

	return tf.metrics.mean(score * 100)