# -*- coding: utf-8 -*-
'''
	LyaBot, Preprocessing
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
	This file is based and inspired by:
		- https://github.com/daniel-kukiela/nmt-chatbot 
		- https://github.com/rsennrich/subword-nmt
'''
import ntpath

import os
import ujson as json
import regex as re
import time
import glob
from multiprocessing import Pool
from itertools import chain
from collections import Counter

from settings import settings
from file_utils import read_lines, write_lines, _LINES_IN_FILE,_FILE_BATCH_SIZE
from vocab import Vocab
from bpe import BPE

class Preprocessing(object):

	def __init__(self, files=[], processes=10, only_data=False):
		if only_data:
			self.files =  [f for f_ in [glob.glob(e) for e in ('{}/static.src'.format(settings.path_data), '{}/static.tgt'.format(settings.path_data))] for f in f_]
		else:
			self.files =  [f for f_ in [glob.glob(e) for e in ('{}/*.src'.format(settings.path_data), '{}/*.tgt'.format(settings.path_data))] for f in f_]
		print(self.files)
		self.processes = 8
		self.bpe = BPE()

		print ('*** Starting Preprocessing on files: \n\t{}'.format('\n\t'.join(self.files)))

	def write_lines(self, file, lines, first_batch):
		if not len(lines) or lines[-1] == '' or lines[-1] == '²':
			lines = list(filter(lambda line: False if line == '' or line == '²' else True, list(lines)))

		file.write(('' if first_batch else '\n') + '\n'.join(lines))	
		return len(lines)


	def learn_bpe(self):
		self.bpe.learn_bpe(self.vocab)

	def create_vocab(self):
		vocab_all = Counter()
		vocab_obj = Vocab()
		for file in self.files:
			print ('*** Starting creating vocab and writting tokenized files {}'.format(file))
			in_path = file
			out_path = in_path.replace(settings.path_data, settings.data_formated).replace(ntpath.basename(in_path), '_tmp_{}'.format(ntpath.basename(in_path)))


			if 'train_0' in in_path:
				out_path_dev = in_path.replace(settings.path_data, settings.data_formated).replace('train_0', '_tmp_dev')
				out_path_test = in_path.replace(settings.path_data, settings.data_formated).replace('train_0', '_tmp_test')
				

			count_lines = 0
			start = time.time()
			written_lines = 0
			test_written = False
			dev_written = False
			lines_in_test_dev = 0
			with open(in_path, 'r', encoding='utf-8', buffering=131072) as in_file:
				with open(out_path, 'w', encoding='utf-8') as out_file:
					with Pool(processes=self.processes) as pool:
						for lines in read_lines(in_file, in_path, int(1e4)):
							count_lines += len(lines)
			
							tokens = pool.map(vocab_obj.tokenizer, lines)

							if 'train_0' in in_path:
								if dev_written == False:
									count_written = 0
									with open(out_path_test, 'w', encoding='utf-8') as out_file_test:
										with open(out_path_dev, 'w', encoding='utf-8') as out_file_dev:
											for token in tokens:
												if test_written == False and count_written < 512:
													written_lines += self.write_lines(out_file_test, [token], (written_lines== 0))
												else:
													if test_written == False:
														test_written = True
														lines_in_test_dev = written_lines
													
													written_lines += self.write_lines(out_file_dev, [token], (written_lines==lines_in_test_dev))

												count_written += 1

									dev_written = True
									lines_in_test_dev = count_written
								else:
									written_lines += self.write_lines(out_file, tokens, (written_lines==lines_in_test_dev))
							else:
								written_lines += self.write_lines(out_file, tokens, (written_lines==0))

							tokens = pool.map(vocab_obj.split, tokens)

							vocab_all.update(chain.from_iterable(tokens))

							print ('\t*** Added to vocab: {}/{} lines ({}s).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)),  end='\r', flush=True)

							if (len(lines) == 0):
								break
						print('\n', flush=True)

		print('\n', flush=True)

		self.generated_vocab = True
		self.vocab = vocab_all
		return vocab_all

	def apply_bpe(self):
		with open(settings.bpe_file, 'r', encoding='utf-8', buffering=131072) as bpe_file:
			self.joins = {tuple(json.loads(k)): v for k, v in json.load(bpe_file).items()}

		files = self.files
		files.append(os.path.join(settings.path_data, 'dev.src'))
		files.append(os.path.join(settings.path_data,'dev.tgt'))
		files.append(os.path.join(settings.path_data,'test.src'))
		files.append(os.path.join(settings.path_data,'test.tgt'))
		out_files = {}

		for file in files:
			print ('*** Applying BPE to {}'.format(file))

			in_path = file.replace(ntpath.basename(file), '_tmp_{}'.format(ntpath.basename(file)))
			in_path = in_path.replace(settings.path_data, settings.data_formated)

			if 'dev.src' in in_path or 'dev.tgt' in in_path or 'test.src' in in_path or 'test.tgt' in in_path:
				out_file = open(file.replace('.src', '.bpe.src').replace('.tgt', '.bpe.tgt').replace(settings.path_data, settings.data_formated), 'w', encoding='utf-8')
				print ('CCC: ', file.replace('.src', '.bpe.src').replace('.tgt', '.bpe.tgt').replace(settings.path_data, settings.data_formated))
			else:
				if '.src' in in_path:
					if 'final.src' not in out_files.keys():
						out_files['final.src'] = open(os.path.join(settings.data_formated, 'final.bpe.src'), 'w', encoding='utf-8')
					out_file = out_files['final.src']
				else:
					if 'final.tgt' not in out_files.keys():
						out_files['final.tgt'] = open(os.path.join(settings.data_formated, 'final.bpe.tgt'), 'w', encoding='utf-8')
					out_file = out_files['final.tgt']

			count_lines = 0
			written_lines = 0
			start = time.time()
			with open(in_path, 'r', encoding='utf-8') as in_file:
				with Pool(processes=self.processes) as pool:
					print ('OOO: ', in_path.replace(settings.path_data, settings.data_formated))

					for lines in read_lines(in_file, in_path, int(1e5)):
						count_lines += len(lines)
						seq = pool.map(self.bpe.apply_bpe_sentence, lines)

						written_lines += self.write_lines(out_file, seq, (written_lines==0))
						print ('\t*** Written to bpe file: {}/{} lines ({}s).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)),  end='\r', flush=True)

						if (len(lines) == 0):
							break
					print('\n', flush=True)

		print('\n', flush=True)



