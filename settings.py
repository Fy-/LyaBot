# -*- coding: utf-8 -*-
'''
	LyaBot, Settings
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

'''

import os
from file_utils import lines_in_file

import tensorflow as tf

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
			- http://files.pushshift.io/reddit/comments/ (Look why they need donations)
)		'''
		self.path_reddit = 'D:\\RED\\2017\\2016' # YOUR REDDIT FOLDER (With .bz2 files)

		''' Raw data (from reddit) and model '''
		self.path_data = os.path.join(self.path, '_data_raw')
		self.path_model = os.path.join(self.path, '_model')
		
		''' Preprocessing '''
		self.path_static = os.path.join(self.path, '_data_static')
		self.protected = self.set_from_file(os.path.join(self.path_static, 'protected.txt'))
		
		''' Vocab & BPE '''
		self.vocab_file = os.path.join(self.path_data, 'vocab.src')
		self.vocab_wanted_size = 20000
		self.data_formated = os.path.join(self.path, '_data_formated')
		self.bpe_file = os.path.join(self.data_formated, 'bpe_joins.json')
		self.vocab_size = self.vocab_wanted_size+3

		''' Training '''
		self.step_per_save = 1000
		self.step_per_show = 100

		''' Model '''
		self.unk = "<unk>" # Same as https://www.youtube.com/watch?v=sSTVECyKlOg
		self.sos = "<s>"
		self.eos = "</s>"
		self.unk_id = 0
		self.sos_id = 1
		self.eos_id = 2
		self.batch_size =  192
		self.num_layers =  2
		self.num_units =  512
		self.learning_rate  = 0.001
		self.max_gradient_norm = 5.0
		self.beam_width = 20
		self.init_weight =  0.1
		self.num_keep_ckpts = 2
		self.dropout = 0.2
		self.num_train_steps = 10000000
		self.num_buckets = 5
		self.epoch_step = 0


		''' It's ugly '''
		if os.path.isfile(os.path.join(self.path_model, 'epoch')):
			with open(os.path.join(self.path_model, 'epoch')) as e:
				self.epoch_step = int(e.readlines()[0])
				
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


settings = Settings()
