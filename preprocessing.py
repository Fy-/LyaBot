# -*- coding: utf-8 -*-
"""
	LyaBot, Preprocessing
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.
"""

import os
import pickle
import json
import html
import regex as re
import glob
from multiprocessing import Pool
from collections import Counter, defaultdict
from tensorflow.python.ops import lookup_ops
from itertools import chain

from settings import settings
from file_utils import read_lines, write_lines, _LINES_IN_FILE

class Preprocessing(object):

	def __init__(self, vocab, files):
		self.regex = {
			'special': re.compile(r'[\x00-\x1f]+|\u3000'),
			'separate': re.compile(r'(?<![▁])([^\w\s\.▁])'),
			'spaces': re.compile(r'[^\S\n]+'),
			'separate_all': re.compile(r'(?<![ ▁])([^ ▁])'),
			'split' : re.compile('(?: |^)(?:▁(▁))?([' + re.escape(r'`~!@#$%^&*()-_=+{[}]:;\'",<>?/|\\') + '0-9]|\.+)'),
			'restorephrases': re.compile(r'P▁R([\d\s▁]+?)P▁R'),
			'restoreperiods': re.compile(r'P▁P([\d\s▁]+?)P▁P'),
			'periods': re.compile('\.{2,}'),
			'protected' : None
		}

		self.magics = {
			'date' : re.compile(r'[0-9]+\/[0-9]+(\/\*\*\*\*|\/[0-9]+)?', re.IGNORECASE),
			'url_1'  : re.compile(r'http[s]?:(//)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE),
			'url_2'  : re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?/[-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&:\\?/~\\#\\%]*[^]'\\.}>\\),\\\"]"),
			'url_3'  : re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?"),
			'email' : re.compile (r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"),
			'facebook' : re.compile(r'(?:(?:http|https):\/\/)?(?:www.)?facebook.com\/(?:(?:\w)*#!\/)?(?:pages\/)?(?:[?\w\-]*\/)?(?:profile.php\?id=(?=\d.*))?([\w\-]*)?'),
			'youtube' : re.compile(r'^(https?\:\/\/)?(www\.)?(youtube\.com|youtu\.?be)\/.+$'),
			'num_digit' : re.compile(r'(\d{1})'),
			'num_small' : re.compile(r'(\d{2})'),
			'num_medium' : re.compile(r'(\d{3,4})'),
			'num_fat' : re.compile(r'(\d{5,10})'),
		}

		self.load_protected()
		self.files = []
		files = glob.glob('{}/*'.format(os.path.join(settings.path_data)))
		for file in files:
			if '.src' in file or '.tgt' in file:
				self.files.append(file)

		print ('Starting Preprocessing on files: {}'.format(str(self.files)))

		self.learned_bpe = False
		self.generated_vocab = False

	@staticmethod
	def create_vocab_tables():
		src_vocab_table = lookup_ops.index_table_from_file(os.path.join(settings.data_formated, 'vocab.bpe.src'), default_value=settings.unk_id, delimiter='\n')
		tgt_vocab_table = src_vocab_table
		return src_vocab_table, tgt_vocab_table

	def replace_url(self, s):
		s = re.sub(self.magics['url_1'],'CALL_RANDOM_URL', s)
		s = re.sub(self.magics['url_2'],'CALL_RANDOM_URL', s)
		s = re.sub(self.magics['url_3'],'CALL_RANDOM_URL', s)

		return s

	def _replace(self, entity):
		phrase = list(filter(None, list(entity.groups())))[0]
		replacement = entity.group(0).replace(phrase, 'P▁R{}P▁R'.format(self.protected_phrases_counter))
		self.protected_phrases_replace.append(phrase)
		self.protected_phrases_counter += 1
		return replacement

	def tokenizer(self, sentence):
		self.protected_phrases_replace = []
		self.protected_phrases_counter = 0
		self.protected_periods_counter = 0

		sentence = sentence.strip()
		sentence = html.unescape(sentence)
		sentence = sentence.replace('<unk>', '').replace('<s>', '').replace('</s>', '').replace('▁','_')

		sentence = self.magics['facebook'].sub('CALL_FACEBOOK', sentence)
		sentence = self.magics['youtube'].sub('CALL_YOUTUBE', sentence)
		sentence = self.replace_url(sentence)
		sentence = self.magics['email'].sub('CALL_EMAIL', sentence)
		sentence = self.magics['date'].sub('CALL_DATE', sentence)

		sentence = self.magics['num_fat'].sub('CALL_NUM_FAT', sentence)
		sentence = self.magics['num_medium'].sub('CALL_NUM_MEDIUM', sentence)
		sentence = self.magics['num_small'].sub('CALL_NUM_SMALL', sentence)
		sentence = self.magics['num_digit'].sub('CALL_NUM_DIGIT', sentence)


		sentence = self.regex['special'].sub(' ', sentence)


		if self.regex['protected'] and self.regex['protected'].search(sentence):
			sentence = self.regex['protected'].sub(self._replace, sentence)

		# Protect multi-periods
		m = self.regex['periods'].findall(sentence)
		if m:
			self.protected_periods_counter += 1
			for dots in sorted(set(m), reverse=True):
				sentence = sentence.replace(dots, 'P▁P{}P▁P'.format(len(dots)))

		sentence = sentence.replace('`', '\'').replace('\'\'', '"')
		sentence = sentence.strip()

		sentence = self.regex['spaces'].sub(' ', sentence)
		sentence = '▁' + sentence.replace(' ', ' ▁')
		sentence = self.regex['separate_all'].sub(' \\1', sentence)

		# Restore protected phrases and multidots
		if self.protected_phrases_counter:
			sentence = self.regex['restorephrases'].sub(lambda number: self.protected_phrases_replace[int(number.group(1).replace(" ", "").replace("▁", ""))], sentence)
		if self.protected_periods_counter:
			sentence = self.regex['restoreperiods'].sub(lambda number: ("." * int(number.group(1).replace(" ", "").replace("▁", ""))), sentence)

		# Replace new line char

		return sentence

	def load_protected(self):
		matched_regexes = []
		unmatched_regexes = []
		phrase = None
		protected_phrase_regex = None
		phrase = re.compile('|'.join(settings.protected))
		print (phrase)

		self.regex['protected'] = phrase if phrase else None

	def write_lines(self, file, lines, first_batch):
		if not len(lines) or lines[-1] == '' or lines[-1] == '²':
			lines = list(filter(lambda line: False if line == '' or line == '²' else True, list(lines)))

		file.write(('' if first_batch else '\n') + '\n'.join(lines))	
		return len(lines)

	def split(self, sentence):
		# Prepare for split sentence into a words by ' ▁'
		line = ' ▁▁' + sentence[1:].replace('▁', '▁▁')
		line = self.regex['split'].sub(r' ▁\1\2 ▁', line)

		# split, filer and return
		return list(filter(lambda line: False if len(line) == 0 or line == '▁' else True, [token.strip() for token in line.split(' ▁')]))

	def apply_bpe(self):
		with open(settings.bpe_file, 'r', encoding='utf-8', buffering=131072) as bpe_file:
			self.joins = {tuple(json.loads(k)): v for k, v in json.load(bpe_file).items()}
			
		files = self.files
		files.append('dev.src')
		files.append('dev.tgt')
		print(files)

		for file in files:
			in_path = os.path.join(settings.data_formated, '_tmp_{}'.format(file))
			out_path = os.path.join(settings.data_formated, file.replace('.src', '.bpe.src').replace('.tgt', '.bpe.tgt'))

			count_lines = 0
			written_lines = 0
			start = time.time()
			with open(in_path, 'r', encoding='utf-8') as in_file:
				with open(out_path, 'w', encoding='utf-8') as out_file:
					with Pool(processes=10) as pool:
						for lines in read_lines(in_file, in_path, int(1e5)):
							count_lines += len(lines)
							seq = pool.map(self.apply_bpe_sentence, lines)

							written_lines += self.write_lines(out_file, seq, (written_lines==0))
							print ('*** Written to bpe file: {}/{} lines ({}).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)))

							if (len(lines) == 0):
								break
		self.learn_bpe = True

	def create_vocab(self):
		vocab_all = Counter()

		for file in self.files:
			in_path = os.path.join(settings.path_data, file)
			out_path = os.path.join(settings.data_formated, '_tmp_{}'.format(file))
			out_path_dev = os.path.join(settings.data_formated, file.replace('train_1', '_tmp_dev'))

			count_lines = 0
			start = time.time()
			written_lines = 0

			with open(in_path, 'r', encoding='utf-8', buffering=131072) as in_file:
				with open(out_path, 'w', encoding='utf-8', buffering=131072) as out_file:
					with open(out_path_dev, 'w', encoding='utf-8') as out_file_dev:
						with Pool(processes=10) as pool:
							for lines in read_lines(in_file, in_path, int(5e4)):
								count_lines += len(lines)
				
								tokens = pool.map(self.tokenizer, lines)

								if written_lines < int(5e4):
									written_lines += self.write_lines(out_file_dev, tokens, (written_lines==0))
								else:
									written_lines += self.write_lines(out_file, tokens, (written_lines==int(5e4)))

								tokens = pool.map(self.split, tokens)

								vocab_all.update(chain.from_iterable(tokens))

								print ('*** Added to vocab: {}/{} lines ({}).'.format(count_lines, _LINES_IN_FILE[in_path], round(time.time() - start, 2)))

								if (len(lines) == 0):
									break

		self.vocab = vocab_all
		return vocab_all

	def apply_bpe_sentence(self, sentence):
		sentence_cache = {}
		joins = self.joins
	
		# Split sentence by ' ▁'
		entities = self.split(sentence)
		new_sentence = []

		# For every entity in sentence
		for entity in entities:

			# If entity exists in cache - used cached (computed earlier) result
			original_entity = entity
			if original_entity in sentence_cache:
				new_sentence.append(sentence_cache[original_entity])
				continue

			# Split entity into pieces (mostly chars)
			entity = entity.split()

			# Make pairs of neighboring pieces/chars
			pairs = []
			prev_char = entity[0]
			for char in entity[1:]:
				pairs.append((prev_char, char))
				prev_char = char

			# Single piece/char - nothing to join
			if not pairs:
				new_sentence.append(entity[0])
				continue

			# Make every possible join
			while True:

				# Joins fragment - includes only pairs that exists in current entity
				subjoins = {pair:joins[pair] for pair in pairs if pair in joins}
				
				# Find most common pair
				pair = min(subjoins, key=subjoins.get, default=())

				# If there's no one - entity is joined
				if not pair or pair not in pairs:
					break

				# prepare pieces/chars
				first, second = pair
				new_pair = first + second

				#print(pairs)

				# Replace every occurence of pair with a joied one
				while pair in pairs:

					# Find pair occurence
					index = pairs.index(pair)

					# Remove pair and update neighbour pairs with joined one
					if index > 0:
						pairs[index - 1] = (pairs[index - 1][0], new_pair)
					if index < len(pairs) - 1:
						pairs[index + 1] = (new_pair, pairs[index + 1][1])
					if len(pairs) == 1:
						pairs[0] = (new_pair, '')
					else:
						del pairs[index]

			# We are going to use first subword from pair to rebuild entity, so we need to add second subword of last entity as a new 'pair'
			# (AB, C), (C, DEF), (DEF, GHIJK) -> AB, C, DEF, GHIJK
			if pairs[-1][1]:
				pairs.append((pairs[-1][1], ''))
			nentity = ' '.join([first for (first, second) in pairs])
			new_sentence.append(nentity)
			sentence_cache[original_entity] = nentity

		# Return joined sentence
		return ' '.join(new_sentence)
		
	def learn_bpe(self, _vocab = None):
		if _vocab == None:
			_vocab = self.vocab


		stats = Counter() # Pair stats
		indices = defaultdict(lambda: defaultdict(int)) # Pairs indexes

		vocab = []
		train_vocab = Counter()
		for i, (entity, freq) in enumerate(_vocab.most_common()):
			# Split vocab token
			entity = tuple(entity.split())

			# Make pairs ("ABCD" -> (A, B), (B, C), (C, D)), stats, indexes and train vocab
			prev_char = entity[0]
			train_vocab[prev_char] += freq
			for char in entity[1:]:
				stats[prev_char, char] += freq
				indices[prev_char, char][i] += 1
				train_vocab[char] += freq
				prev_char = char
			vocab.append((entity, freq))


		print ('Learning BPE for vocab of {} tokens'.format(settings.vocab_wanted_size))

		# List of joins per vocab
		joins= []

		# Partial stats speeds up learning process - optimization for 'max' above
		partial_stats = Counter(['', -1])
		partial_stats_min = -1
		update_partial_stats = True

		# Current number of vocab tokens
		train_vocab_len = prev_train_vocab_len = len(train_vocab)

		# Learn until vocab will contain desired number of tokens
		while train_vocab_len < settings.vocab_wanted_size:
		    print('*** BPE {}/{}'.format(prev_train_vocab_len, settings.vocab_wanted_size), end='\r', flush=True) 

		    clean_train_vocab = False

		    # Get most frequent pair
		    most_frequent, freq = partial_stats.most_common(1)[0]

		    # Update partial stats or frequency of most frequent pair is less than saved minimum for partial stats
		    if update_partial_stats or freq < partial_stats_min:
		        partial_stats_min = stats.most_common(500)[-1][1]
		        partial_stats = Counter()
		        for k, v in stats.most_common():
		            if v < partial_stats_min:
		                break
		            partial_stats[k] = v
		        update_partial_stats = False

		        # Get most frequent pair (again, proper one this time)
		        most_frequent, _ = partial_stats.most_common(1)[0]

		    # If frequency is lower than 2 - exit
		    if stats[most_frequent] < 2:
		        print('No pair has frequency greater than 1. Stopping earlier, your vocab file will include less tokens.\n')
		        break

		    # Replace pair "A B" with new entity "AB"

		    # Changes made
		    changes = []

		    # Replace regex
		    pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(most_frequent)) + r'(?!\S)')

		    # Loop through indices
		    for j, freq in indices[most_frequent].items():

		        # Do not touch not existent pairs
		        if freq < 1:
		            continue

		        # Get entity and frequency
		        entity, freq = vocab[j]

		        # Replace "A B" with "AB" in entity
		        new_entity = pattern.sub(''.join(most_frequent), ' '.join(entity))
		        new_entity = tuple(new_entity.split())

		        # Update entity
		        vocab[j] = (new_entity, freq)

		        changes.append((j, new_entity, entity, freq))

		    # Update indices and pair stats
		    # Merged pair doesn't exist anymore
		    stats[most_frequent] = 0
		    partial_stats[most_frequent] = 0
		    indices[most_frequent] = defaultdict(int)

		    # Get entities and a new pair
		    first, second = most_frequent
		    new_pair = first + second

		    # Iterate through all changes
		    for j, entity, old_entity, freq in changes:

		        # Find all occurences of first pair entity
		        prev = -2
		        for i in iter([i for i, entity in enumerate(old_entity) if entity == first]):

		            # Do not touch second "B B" if "B B B"
		            if i == prev + 1:
		                continue

		            # Check if second pair entity follows first one
		            if i < len(old_entity) - 1 and old_entity[i + 1] == second:

		                # Reduce frequency of "A B" in "A B C D" where "B C" is a merged pair
		                if i:
		                    prev = old_entity[i - 1:i + 1]
		                    stats[prev] -= freq
		                    partial_stats[prev] = stats[prev]
		                    indices[prev][j] -= 1

		                # Reduce frequency of "C D" in "A B C D" where "B C" is a merged pair
		                if i < len(old_entity) - 2:

		                    # But do not touch "C B" if "A B C B C" as values will be adjusted with next occurence of "B C" pair
		                    if old_entity[i + 2] != first or i >= len(old_entity) - 3 or old_entity[i + 3] != second:
		                        next = old_entity[i + 1:i + 3]
		                        stats[next] -= freq
		                        partial_stats[next] = stats[next]
		                        indices[next][j] -= 1

		                prev = i

		                if train_vocab[first] <= freq or train_vocab[second] <= freq:
		                    clean_train_vocab = True
		                train_vocab[first] -= freq
		                train_vocab[second] -= freq

		        # Find all occurences of first pair entity
		        for i in [i for i, entity in enumerate(entity) if entity == new_pair]:

		            # Increase frequency of (new pair) "A BC" in "A BC D"
		            if i:
		                prev = entity[i - 1:i + 1]
		                stats[prev] += freq
		                #if stats[prev] >= partial_stats_min:
		                #    update_partial_stats = True
		                partial_stats[prev] = stats[prev]
		                indices[prev][j] += 1

		            # Increase frequency of (new pair) "BC D" in "A BC D", but do not touch if "A BC BC" as stats for "BC BC" will be adjusted win next occurence of "BC" pair
		            if i < len(entity) - 1 and entity[i + 1] != new_pair:
		                next = entity[i:i + 2]
		                stats[next] += freq
		                #if stats[next] >= partial_stats_min:
		                #    update_partial_stats = True
		                partial_stats[prev] = stats[prev]
		                indices[next][j] += 1

		            # Set frequency of a new pair
		            train_vocab[new_pair] += freq

		    # Current pair is merged - is not a pair anymore, so has frequency of 0
		    stats[most_frequent] = 0
		    partial_stats[most_frequent] = 0

		    # Remove (from training vocab) tokens with frequency of 0
		    if clean_train_vocab:
		        train_vocab= +train_vocab

		    # Calculate current number of train vocab entities
		    prev_train_vocab_len = train_vocab_len
		    train_vocab_len = len(train_vocab)
		    train_vocab_len_diff = train_vocab_len - prev_train_vocab_len


		    # Add new join pair
		    joins.append(most_frequent)

		# Save list of joins for train vocab
		joins = dict(reversed([(v, i) for i, v in enumerate(joins)]))

		with open(settings.bpe_file, 'w', encoding='utf-8', buffering=131072) as bpe_file:
			print('SAVED')
			json.dump({json.dumps(k):v for k,v in joins.items()}, bpe_file)

		data_vocab = [entity for entity, _ in train_vocab.most_common()]
		with open(os.path.join(settings.data_formated, 'vocab.bpe.src'), 'w', encoding='utf-8')  as vocab_file:
			vocab_file.write(('\n'.join([settings.unk, settings.sos, settings.eos]) + '\n' + "\n".join(data_vocab[:settings.vocab_max_size])))

		self.joins = joins
		self.generate_vocab = True
