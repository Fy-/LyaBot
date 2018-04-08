# -*- coding: utf-8 -*-
'''
	LyaBot, Settings
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
import ujson as json
import tensorflow as tf
from file_utils import lines_in_file

class Settings(object):
	def __init__(self):
		self.path = os.path.realpath(os.path.dirname(__file__) )

		''' Reddit datas 
			If you downloaded reddit exports in .xz format, just import lzma.
			Replace bz2.open by lzma.open il reddit_import, it will work fine ^^
			You can download reddit comments from here for example:
			- 2015 : https://cinnamon.dewarim.com/torrents/reddit-2015.torrent (68gb)
			- 2016 : https://cinnamon.dewarim.com/torrents/reddit-2016.torrent (81gb)
			- 2017 : https://cinnamon.dewarim.com/torrents/reddit-2017.torrent (23gb (only 3 month))
			- http://files.pushshift.io/reddit/comments/ (please consider making a donation)
)		'''
		self.path_reddit = 'D:\\RED' # YOUR REDDIT FOLDER (With .bz2 files)

		''' Raw data (from reddit) and model '''
		self.path_data = os.path.join(self.path, '_data')
		self.path_model = os.path.join(self.path, '_model')
		self.path_best_accuracy = os.path.join(self.path_model, 'best_accuracy')
		self.path_best_word_accuracy = os.path.join(self.path_model, 'best_accuracy_words')
		self.path_best_bleu = os.path.join(self.path_model, 'best_bleu')

		''' Preprocessing '''
		self.path_static = os.path.join(self.path, '_data_static')
		self.protected = self.set_from_file(os.path.join(self.path_static, 'protected.txt'))
		
		''' Vocab & BPE '''
		self.vocab_file = os.path.join(self.path_data, 'vocab.src')
		self.vocab_wanted_size = 40000
		self.data_formated = os.path.join(self.path, '_data_formated')
		self.bpe_file = os.path.join(self.data_formated, 'bpe_joins.json')
		self.vocab_size = self.vocab_wanted_size+3

		''' Training '''
		self.step_per_save = 1000
		self.step_per_show = 100
		self.step_per_eval = 2000

		''' Model '''
		self.unk = "<unk>" # Same as https://www.youtube.com/watch?v=sSTVECyKlOg
		self.sos = "<s>"
		self.eos = "</s>"
		self.unk_id = 0
		self.sos_id = 1
		self.eos_id = 2
		self.batch_size =  96
		self.num_layers =  2
		self.num_units =  512	
		self.learning_rate  = [0.001, 0.001, 0.0001]
		self.max_gradient_norm = 5.0
		self.beam_width = 20
		self.init_weight =  0.1
		self.num_keep_ckpts = 2
		self.dropout = 0.2
		self.num_train_steps = 10000000
		self.num_buckets = 5

		self.load()

	def save(self):
		obj = {
			'epoch' : self.epoch,
			'epoch_step': self.epoch_step,
			'best_accuracy' : self.best_accuracy,
			'best_bleu' : self.best_bleu,
			'best_word_accuracy' : self.best_word_accuracy
		}
		with open(os.path.join(self.path_model, 'settings.json'), 'w', encoding='utf-8') as e:
			e.write(json.dumps(obj))

	def load(self):
		if os.path.isfile(os.path.join(self.path_model, 'settings.json')):
			with open(os.path.join(self.path_model, 'settings.json'), 'r', encoding='utf-8') as e:
				obj = json.loads(e.readlines()[0])

				for k, v in obj.items():
					setattr(self, k, v)
		else:
			self.epoch = 0
			self.epoch_step = 0
			self.best_accuracy = 0
			self.best_bleu = 0
			self.best_word_accuracy = 0
				
	def set_from_file(self, file):
		''' Just a tool to set a variable from a text file, for example, a regex list. 
			lines who starts with # will be ignored.
			@TODO: I don't remember.
		'''
		if not os.path.isfile(file):
			print ("!*** {} not found.".format(file))
			output = []
		else:
			with open(file, 'r', encoding='utf-8') as f:
				output = list(filter(lambda word: False if word[0] == '#' else True, filter(None, f.read().split("\n"))))
		return output

	def get_config_proto(self, log_device_placement=False, allow_soft_placement=True, num_intra_threads=0, num_inter_threads=0):
		config_proto = tf.ConfigProto(
			log_device_placement=log_device_placement,
			allow_soft_placement=allow_soft_placement
		)

		config_proto.gpu_options.allow_growth = True
		if num_intra_threads:
			config_proto.intra_op_parallelism_threads = num_intra_threads
		if num_inter_threads:
			config_proto.inter_op_parallelism_threads = num_inter_thread

		return config_proto
settings = Settings()
