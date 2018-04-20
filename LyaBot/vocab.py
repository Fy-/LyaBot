# -*- coding: utf-8 -*-
'''
	LyaBot, Vocab
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

import regex as re
import html
import os

from collections import Counter
from tensorflow.python.ops import lookup_ops
from .settings import settings

class Vocab(object):
	def __init__(self):
		self.regex = {
			'special': re.compile(r'[\x00-\x1f]+|\u3000'),
			'separate': re.compile(r'(?<![▁])([^\w\s\.▁])'),
			'spaces': re.compile(r'[^\S\n]+'),
			'separate_all': re.compile(r'(?<![ ▁])([^ ▁])'),
			'split' : re.compile('(?: |^)(?:▁(▁))?([' + re.escape(r'`~!@#$%^&*()-_=+{[}]:;\'",<>?/|\\') + '0-9]|\.+)'),
			'restorephrases': re.compile(r'P▁R([\d\s▁]+?)P▁R'),
			'restoreperiods': re.compile(r'P▁P([\d\s▁]+?)P▁P'),
			'periods': re.compile(r'\.{2,}'),
			'protected' : None,
			'multiples' :  re.compile(r'(.)(\1{4,})')
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

	def replace_url(self, s):
		s = re.sub(self.magics['url_1'],'CALL_RANDOM_URL', s)
		s = re.sub(self.magics['url_2'],'CALL_RANDOM_URL', s)
		s = re.sub(self.magics['url_3'],'CALL_RANDOM_URL', s)

		return s

	def split(self, sentence):
		# Prepare for split sentence into a words by ' ▁'
		line = ' ▁▁' + sentence[1:].replace('▁', '▁▁')
		line = self.regex['split'].sub(r' ▁\1\2 ▁', line)

		# split, filer and return
		return list(filter(lambda line: False if len(line) == 0 or line == '▁' else True, [token.strip() for token in line.split(' ▁')]))

	def tokenizer(self, sentence):
		self.protected_phrases_replace = []
		self.protected_phrases_counter = 0
		self.protected_periods_counter = 0

		sentence = sentence.strip()
		sentence = html.unescape(sentence)
		sentence = self.regex['multiples'].sub(r'\1\1\1\1\1', sentence)
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

		self.regex['protected'] = phrase if phrase else None

	@staticmethod
	def create_vocab_tables():
		src_vocab_table = lookup_ops.index_table_from_file(os.path.join(settings.data_formated, 'vocab.bpe.src'), default_value=settings.unk_id, delimiter='\n')
		tgt_vocab_table = src_vocab_table
		return src_vocab_table, tgt_vocab_table

	def _replace(self, entity):
		phrase = list(filter(None, list(entity.groups())))[0]
		replacement = entity.group(0).replace(phrase, 'P▁R{}P▁R'.format(self.protected_phrases_counter))
		self.protected_phrases_replace.append(phrase)
		self.protected_phrases_counter += 1
		return replacement
