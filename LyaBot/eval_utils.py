# -*- coding: utf-8 -*-
'''
	LyaBot, Eval utils
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
	This file is inspired by https://github.com/tensorflow/nmt
'''
import os
import random
import nltk
import tensorflow as tf 
import time
import math
from .model_utils import create_or_load_model
from .file_utils import lines_in_file, load_data_readlines
from .settings import settings
from .train_utils import add_summary

def compute_perplexity(model, sess, name):
	def safe_exp(value):
		''' Exponentiation with catching of overflow error. (And not The Barrens) '''
		try:
			ans = math.exp(value)
		except OverflowError:
			ans = float("inf")
		return ans	

	total_loss = 0
	total_predict_count = 0
	start_time = time.time()

	while True:
		try:
			loss, predict_count, batch_size = model.eval(sess)
			total_loss += loss * batch_size
			total_predict_count += predict_count
		except tf.errors.OutOfRangeError:
			break	

	perplexity = safe_exp(total_loss / total_predict_count)

	print ('*** Eval perplexity: {:.2f}'.format( perplexity))
	return perplexity

def run_infer_sample(infer_model, infer_sess, src_data, tgt_data, n=1):
	''' Because training is boring '''
	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, 'infer')

	for i in range(n):
		decode_id = random.randint(0, len(src_data) - 1)

		iterator_feed_dict = {
			infer_model.src_placeholder: [src_data[decode_id]],
			infer_model.batch_size_placeholder: 1,
		}	

		infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
		nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)

		nmt_outputs = nmt_outputs[0] 

		seq = get_sentence(nmt_outputs)
		print('\n\t*** RedditInput  : {}'.format(
				format_spm_text(
					src_data[decode_id].encode('utf-8').split()
				).decode('utf-8')
			)
		)
		#print('*** target_data : __{}__'.format(' '.join(tgt_data[decode_id]))
		print('\t*** LyaNMT v0.1a : {}'.format(seq.decode('utf-8')))

	print('\n')

def get_sentence(nmt_outputs, sent_id=0):
	output = nmt_outputs[sent_id, :].tolist()
	eos = settings.eos.encode('utf-8')
	if eos in output:
		output = output[:output.index(eos)]

	return format_spm_text(output)

def get_sentence_tokenized(nmt_outputs, sent_id=0):
	output = nmt_outputs[sent_id, :].tolist()
	eos = settings.eos.encode('utf-8')
	if eos in output:
		output = output[:output.index(eos)]

	return format_text(output)

def format_text(words):
	''' Convert a sequence words into sentence. '''
	if (not hasattr(words, '__len__') and  # for numpy array
		not isinstance(words, collections.Iterable)):
		words = [words]

	return b" ".join(words)

def format_spm_text(symbols):
	return u"".join(format_text(symbols).decode('utf-8').split()).replace(u'\u2581', u' ').strip().encode('utf-8')

def format_spm_text_str(symbols):
	words = ' '.join(symbols)
	return u"".join(words.split()).replace(u'\u2581', u' ').strip()

def run_eval(eval_model, eval_sess, summary_writer):
	with eval_model.graph.as_default():
		loaded_eval_model, global_step = create_or_load_model(eval_model.model, eval_sess, 'eval')

	src_file = os.path.join(settings.data_formated, 'dev.bpe.src')
	tgt_file = os.path.join(settings.data_formated, 'dev.bpe.tgt')	


	iterator_feed_dict = {
		eval_model.src_file_placeholder: src_file,
		eval_model.tgt_file_placeholder: tgt_file
	}

	eval_sess.run(eval_model.iterator.initializer, feed_dict=iterator_feed_dict)
	ppl = compute_perplexity(eval_model.model, eval_sess, 'eval')
	add_summary(summary_writer, global_step, 'eval_ppl', ppl)

	return ppl

def run_full_eval(infer_model, infer_sess, summary_writer, save_best=True):
	print ('*** Starting EVAL on test.bpe.src ...\n')
	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, 'infer')

	src_file = os.path.join(settings.data_formated, 'test.bpe.src')
	tgt_file = os.path.join(settings.data_formated, 'test.bpe.tgt')

	iterator_feed_dict = {
		infer_model.src_placeholder: load_data_readlines(src_file),
		infer_model.batch_size_placeholder: 128,
	}

	infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
	output = os.path.join(settings.path_model, 'output_eval')
	scores = decode_and_evaluate(infer_model.model, infer_sess, output, tgt_file)

	for metric in scores.keys():
		print('\t {} : {}'.format(metric, scores[metric]))
		best_metric_label = 'best_' + metric
		add_summary(summary_writer, global_step, metric, scores[metric])

		if save_best and scores[metric] > getattr(settings, best_metric_label):
			setattr(settings, best_metric_label, scores[metric])
			infer_model.model.saver.save(infer_sess, os.path.join(getattr(settings, 'path_' + best_metric_label), 'model.ckpt'), global_step=global_step)
	settings.save()
	print('\n')

def decode_and_evaluate(model, sess, output, ref_file):
	with open(output, mode='w', encoding='utf-8') as reply_file:
		while True:
			try:
				nmt_outputs, _ = model.decode(sess)
				batch_size = nmt_outputs.shape[1]

				for sent_id in range(batch_size):
					reply = get_sentence_tokenized(nmt_outputs[0], sent_id)
					reply_file.write((reply + b'\n').decode('utf-8'))

			except tf.errors.OutOfRangeError:
				break

	evaluation_scores = {}
	try:
		evaluation_scores['bleu'] = _bleu(ref_file, output)
	except:
		evaluation_scores['bleu'] = 0.

	try:
		evaluation_scores['accuracy'] = _accuracy(ref_file, output)
	except:
		evaluation_scores['accuracy'] = 0.

	try:
		evaluation_scores['word_accuracy'] = _word_accuracy(ref_file, output)
	except:
		evaluation_scores['word_accuracy'] = 0.

	return evaluation_scores

def _bleu(label_path, output_path):
	cc = nltk.translate.bleu_score.SmoothingFunction()
	with open(label_path, 'r', encoding='utf-8') as label_file:
		with open(output_path, 'r', encoding='utf-8') as pred_file:
			score = 0.
			count = 0.
			for sentence in label_file:
				labels = format_spm_text_str(sentence.strip().split(" ")).split(' ')
				preds = format_spm_text_str(pred_file.readline().strip().split(" ")).split(' ')
				score += nltk.translate.bleu_score.sentence_bleu([labels], preds, smoothing_function=cc.method4)
				count += 1

	return float(round((score / count), 2))

def _word_accuracy(label_path, output_path):
	with open(label_path, 'r', encoding='utf-8') as label_file:
		with open(output_path, 'r', encoding='utf-8') as pred_file:
			acc, count = 0., 0.
			for sentence in label_file:
				labels = format_spm_text_str(sentence.strip().split(" ")).split(' ')
				preds = format_spm_text_str(pred_file.readline().strip().split(" ")).split(' ')
				match = 0.
				for pos in range(min(len(labels), len(preds))):
					label = labels[pos]
					pred = preds[pos]
					if label == pred:
						match += 1

				acc += 100 * match / max(len(labels), len(preds))
				count += 1
	return float(round((acc / count), 2))

def _accuracy(label_path, output_path):
	with open(label_path, 'r', encoding='utf-8') as label_file:
		with open(output_path, 'r', encoding='utf-8') as pred_file:
			count = 0.
			match = 0.
			for label in label_file:
				label = format_spm_text_str(label.strip().split(' '))
				pred = format_spm_text_str(pred_file.readline().strip().split(' '))

				if label == pred:
					match += 1

				count += 1

	return float(round((100 * match / count), 2))

