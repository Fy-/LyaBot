# -*- coding: utf-8 -*-
'''
	LyaBot, RNJesus utilis
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

'''
import codecs
import math
import random
import tensorflow as tf

from settings import settings
from model_utils import create_or_load_model

def format_text(words):
	''' Convert a sequence words into sentence. '''
	if (not hasattr(words, "__len__") and  # for numpy array
		not isinstance(words, collections.Iterable)):
		words = [words]

	return b" ".join(words)

def format_spm_text(symbols):
	''' Decode a text in SPM (https://github.com/google/sentencepiece) format. '''
	return u"".join(format_text(symbols).decode("utf-8").split()).replace(u"\u2581", u" ").strip().encode("utf-8")

def load_data(inference_input_file):
	''' This is the most usefull function of all time in all universe. Seriously, need to remove this '''
	with codecs.getreader("utf-8")(tf.gfile.GFile(inference_input_file, mode="rb")) as f:
		inference_data = f.read().splitlines()
	return inference_data

def get_sentence(nmt_outputs):
	output = nmt_outputs[0, :].tolist()
	eos = settings.eos.encode("utf-8")
	if eos in output:
		output = output[:output.index(eos)]

	return format_spm_text(output)

def run_infer_sample(infer_model, infer_sess, src_data, tgt_data, n=1):
	''' Because training is boring '''
	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, "infer")

	for i in range(n):
		decode_id = random.randint(0, len(src_data) - 1)

		iterator_feed_dict = {
			infer_model.src_placeholder: [src_data[decode_id]],
			infer_model.batch_size_placeholder: 1,
		}	

		infer_sess.run(infer_model.iterator.initializer, feed_dict=iterator_feed_dict)
		nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)

		nmt_outputs = nmt_outputs[0] # beam

		seq = get_sentence(nmt_outputs)
		print('\n\t*** RedditInput  : {}'.format(
				format_spm_text(
					src_data[decode_id].encode('utf-8').split()
				).decode('utf-8')
			)
		)
		#print('*** target_data : __{}__'.format(' '.join(tgt_data[decode_id]))
		print('\t*** LyaNMT v0.1a : {}'.format(seq.decode("utf-8")))

	print('\n')

def safe_exp(value):
	''' Exponentiation with catching of overflow error. (And not The Barrens) '''
	try:
		ans = math.exp(value)
	except OverflowError:
		ans = float("inf")
	return ans