# -*- coding: utf-8 -*-
'''
	LyaBot, Preprocessing
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

	This file is based and inspired by:
		- https://github.com/daniel-kukiela/nmt-chatbot 
		- https://github.com/rsennrich/subword-nmt
'''

import os
import ujson as json
import regex as re
import time
from multiprocessing import Pool
from itertools import chain
from collections import Counter


from settings import settings
from file_utils import read_lines, write_lines, _LINES_IN_FILE,_FILE_BATCH_SIZE
from vocab import Vocab
from bpe import BPE

class Preprocessing(object):

	def __init__(self, files, processes=10):
		self.files = files
		self.processes = processes
		self.bpe = BPE()
		print ('Starting Preprocessing on files: {}'.format(str(self.files)))

	def write_lines(self, file, lines, first_batch):
		if not len(lines) or lines[-1] == '' or lines[-1] == '²':
			lines = list(filter(lambda line: False if line == '' or line == '²' else True, list(lines)))

		file.write(('' if first_batch else '\n') + '\n'.join(lines))	
		return len(lines)


	def learn_bpe(self):
		self.bpe.learn_bpe(self.vocab)

	def create_vocab(self):
		vocab_all = Counter()
		voc_obj = Vocab()
		for file in self.files:
			print ('*** Starting creating vocab and tokenizing {}'.format(file))
			in_path = os.path.join(settings.path_data, file)
			out_path = os.path.join(settings.data_formated, '_tmp_{}'.format(file))

			out_path_dev = os.path.join(settings.data_formated, file.replace('train_1', '_tmp_dev'))
				

			count_lines = 0
			start = time.time()
			written_lines = 0

			with open(in_path, 'r', encoding='utf-8', buffering=131072) as in_file:
				with open(out_path, 'w', encoding='utf-8') as out_file:
					with open(out_path_dev, 'w', encoding='utf-8') as out_file_dev:
						with Pool(processes=self.processes) as pool:
							for lines in read_lines(in_file, in_path, int(4e4)):
								count_lines += len(lines)
				
								tokens = pool.map(voc_obj.tokenizer, lines)

								if 'train' in in_path:
									if written_lines < _FILE_BATCH_SIZE[in_path]:
										written_lines += self.write_lines(out_file_dev, tokens, (written_lines==0))
									else:
										written_lines += self.write_lines(out_file, tokens, (written_lines==int(_FILE_BATCH_SIZE[in_path])))
								else:
									written_lines += self.write_lines(out_file, tokens, (written_lines==0))

								tokens = pool.map(voc_obj.split, tokens)

								vocab_all.update(chain.from_iterable(tokens))

								print ('\t*** Added to vocab: {}/{} lines ({}s).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)),  end='\r', flush=True)

								if (len(lines) == 0):
									break

		print('\n', end='\r', flush=True)

		self.vocab = vocab_all
		return vocab_all

	def apply_bpe(self):
		with open(settings.bpe_file, 'r', encoding='utf-8', buffering=131072) as bpe_file:
			self.joins = {tuple(json.loads(k)): v for k, v in json.load(bpe_file).items()}

		files = self.files
		files.append('dev.src')
		files.append('dev.tgt')

		for file in files:
			print ('*** Applying BPE to {}'.format(file))

			in_path = os.path.join(settings.data_formated, '_tmp_{}'.format(file))
			out_path = os.path.join(settings.data_formated, file.replace('.src', '.bpe.src').replace('.tgt', '.bpe.tgt'))

			count_lines = 0
			written_lines = 0
			start = time.time()
			with open(in_path, 'r', encoding='utf-8') as in_file:
				with open(out_path, 'w', encoding='utf-8') as out_file:
					with Pool(processes=self.processes) as pool:
						for lines in read_lines(in_file, in_path, int(1e5)):
							count_lines += len(lines)
							seq = pool.map(self.bpe.apply_bpe_sentence, lines)

							written_lines += self.write_lines(out_file, seq, (written_lines==0))
							print ('\t*** Written to bpe file: {}/{} lines ({}s).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)),  end='\r', flush=True)

							if (len(lines) == 0):
								break

		print('\n', end='\r', flush=True)



