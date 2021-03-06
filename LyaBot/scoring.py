# -*- coding: utf-8 -*-
'''
	LyaBot, Scoring
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
from difflib import SequenceMatcher
from collections import OrderedDict
import regex as re


class Scoring(object):
	_poncts = re.compile(r'[\.!\?;]$') # for end lines
	_emotes_utf8 = re.compile(r'[\u263a-\U0001f645]') # smileyyyy!
	_emotes = [':)', ':(', ':D', ':P', ':p', '^^', '^_^', 'o_O', 'O_o', ':o', ':/', '/o/'] # oO

	_subsentences = [
		# (score, question, answer)
		(-2, None, re.compile('(?i)I don\'t know')),
		(-2, None, re.compile('(?i)I don\'t think')),
		(-10, None, re.compile('(?i)no, no, no, no')),
		(-10, None, re.compile('(?i)yes, yes, yes, yes')),
		(-15, None, re.compile('(?i)dozens of us')),
		(3, re.compile('(?i)you( are|\'re)|are you'), re.compile('(?i) I\'m')),
		(3, re.compile('(?i)how old'), re.compile('(?i)years old|CALL_NUM_')),
		(3, re.compile('(?i)what color'), re.compile('(?i)blue|yellow|red|orange|grey|green|pink|black|white|turquoise')), # color list plz
		(3, re.compile('(?i)how many'), re.compile('CALL_NUM_'))
	]

	def __init__(self, question, _awnsers):

		awnsers =[]
		for awnser in _awnsers:
			if awnser not in awnsers:
				awnsers.append(awnser)

		self.question = question
		self.index = question
		self.awnsers = {}
		self.scores = {}

		index = 0
		for awnser in awnsers:
			self.scores[index] = self.check_score(awnser, index)

			self.awnsers[index] = awnser
			index += 1

	def get_best_scores(self, m):
		scores = sorted(self.scores.items(), key=lambda x: x[1], reverse=True)
		result = {}
		i = 0
		for index, score in scores:
			result[i] = self.awnsers[index]
			i += 1
			if i >= m:
				break

		return result

	def check_score(self, awnser, index):
		score = 15

		# use default score
		if index in [0,1]: 
			score += 2
		if index in [2,3]:
			score += 1
		if index in [4,5]:
			score += .5

		score += self.check_ending(awnser[-1:len(awnser)]) # last char checkings
		score += self.check_subsentence(awnser) # check _subsentences
		score += self.check_smilarity(awnser) # check similarity
		return score

	def check_subsentence(self, awnser):
		score = 0
		for score, q, a in Scoring._subsentences:
			if (q == None or re.search(q, self.question)) and re.search(a, awnser):
				score += score
		return score

	def check_smilarity(self, awnser):
		score = 0
		similar = SequenceMatcher(None, awnser, self.question).ratio()
		if similar > .8:
			score = -2
		if similar < .35:
			score = 1

		return score

	def check_ending(self, ending):
		if re.search(Scoring._poncts, ending) or re.search(Scoring._emotes_utf8, ending) or ending in Scoring._emotes:
			return 10
		else:
			return -10
