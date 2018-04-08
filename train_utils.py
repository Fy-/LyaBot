# -*- coding: utf-8 -*-
'''
	LyaBot, Train utils
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

import nltk
import math
from time import time
import tensorflow as tf
import os

from settings import settings
from file_utils import lines_in_file

def add_summary(summary_writer, global_step, tag, value):
	''' Add a new summary to the current summary_writer. '''
	summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
	summary_writer.add_summary(summary, global_step)

class TrainStats(object):
	def __init__(self, train_model):
		self.step_time = 0.0
		self.loss = 0.0
		self.predict_count = 0.0
		self.total_count = 0.0
		self.grad_norm = 0.0
		self.summary_writer = tf.summary.FileWriter(os.path.join(settings.path_model, 'log_dir/'), train_model.graph)

	def reset(self):
		self.step_time = 0.0
		self.loss = 0.0
		self.predict_count = 0.0
		self.total_count = 0.0
		self.grad_norm = 0.0

	def update(self, start_time, step_result):
		(_, step_loss, step_predict_count, step_summary, global_step, step_word_count, batch_size, grad_norm, learning_rate) = step_result
		self.step_time += (time() - start_time)
		self.loss += (step_loss * batch_size)
		self.predict_count += step_predict_count
		self.total_count += float(step_word_count)
		self.grad_norm += grad_norm
		self.global_step = global_step
		self.learning_rate = learning_rate
		self.summary_writer.add_summary(step_summary, global_step)
		return global_step

	def process(self, global_step, step_per_stats):
		def safe_exp(value):
			''' Exponentiation with catching of overflow error. (And not The Barrens) '''
			try:
				ans = math.exp(value)
			except OverflowError:
				ans = float("inf")
			return ans

		self.info_step_time = self.step_time / step_per_stats
		self.info_grad_norm = self.grad_norm / step_per_stats
		self.info_train_ppl = safe_exp(self.loss / self.predict_count)
		self.info_speed = self.total_count / (1000 * self.step_time)
		is_overflow = False
		if math.isnan(self.info_train_ppl) or math.isinf(self.info_train_ppl) or self.info_train_ppl > 1e20:
			is_overflow = True
		return is_overflow

	def print_step_info(self):
		print ('*** STEP {}'.format(self.global_step))
		print ('\t Learning rate: {}'.format(self.learning_rate))
		print ('\t Step time (avg): {:2f} | Speed: {:2f}'.format(self.info_step_time, self.info_speed))
		print ('\t Train PPL: {:2f} | Grad Norm (avg) {:2f}'.format(self.info_train_ppl, self.info_grad_norm))
		print ('\n')



	def __del__(self):
		print ('*** Finished an epoch, saving datas.')
		self.summary_writer.close()
		settings.save()

def calc_num_steps(e):
	csize = 0
	for file in ['final.bpe.src']:
		csize += lines_in_file(os.path.join(settings.data_formated, file))

	steps = math.ceil((e+1) * csize / (settings.batch_size))
	return steps

def blue_score(labels, predictions):

	#nltk.translate.bleu_score

	return tf.metrics.mean(score * 100)