# -*- coding: utf-8 -*-
'''
	LyaBot, Replies
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
import time
import datetime as dt
import calendar as cal

from settings import settings
from bpe import BPE
from vocab import Vocab
from eval_utils import format_spm_text, get_sentence, run_infer_sample
from scoring import Scoring
from random import randint



class Reply(object):
	def __init__(self, infer_model, infer_sess, loaded_infer_model):
		self.infer_model = infer_model
		self.infer_sess = infer_sess
		self.loaded_infer_model = loaded_infer_model
		self.start = time.time()
		self.default_static_src = []
		with open(os.path.join(settings.path_static, 'default.src')) as  f: 
			for l in f.readlines():
				self.default_static_src.append(l.replace('\n', ''))

		self.default_static_tgt = []
		with open(os.path.join(settings.path_static, 'default.tgt')) as  f: 
			for l in f.readlines():
				self.default_static_tgt.append(l.replace('\n', ''))	

		self.bpe = BPE()
		self.vocab = Vocab()

	@staticmethod
	def get_weekday(day_delta):
		# https://github.com/bshao001/ChatLearner/blob/master/chatbot/functiondata.py
		now = dt.datetime.now()
		if day_delta == 'd_2':
			day_time = now - dt.timedelta(days=2)
		elif day_delta == 'd_1':
			day_time = now - dt.timedelta(days=1)
		elif day_delta == 'd1':
			day_time = now + dt.timedelta(days=1)
		elif day_delta == 'd2':
			day_time = now + dt.timedelta(days=2)
		else:
			day_time = now

		weekday = cal.day_name[day_time.weekday()]
		return "{}, {:%B %d, %Y}".format(weekday, day_time)

	def replace(self, s):
		s = s.replace('CALL_GET_NAME', 'Lya')
		s = s.replace('CALL_GET_MYFULLNAME', 'Lya v0.1')
		s = s.replace('CALL_GET_MYURL', 'https://github.com/Fy-/LyaBot')
		s = s.replace('CALL_GET_MYLOCATION', 'Montpellier (France)')
		s = s.replace('CALL_MYAGE', '0.1')
		s = s.replace('CALL_MYUPTIME', str(time.time()-self.start) + ' seconds')
		s = s.replace('CALL_NUM_DIGIT', str(randint(0, 9)))
		s = s.replace('CALL_NUM_SMALL', str(randint(15, 80)))
		s = s.replace('CALL_NUM_MEDIUM', str(randint(100, 900)))
		s = s.replace('CALL_GET_TIME', str(time.strftime("%I:%M %p")))
		s = s.replace('CALL_GET_TODAY_DATE', "{:%B %d, %Y}".format(dt.date.today()))
		s = s.replace('CALL_GET_TODAY_DAY', self.get_weekday('d_0'))
		s = s.replace('CALL_GET_YESTERDAY_DATE', self.get_weekday('d_1'))
		s = s.replace('CALL_GET_TOMORROW_DAY', self.get_weekday('d1'))
		s = s.replace('CALL_GET_TOMORROW_DATE', self.get_weekday('d1'))
		s = s.replace('CALL_EMAIL', 'm@fy.to')
		return s

	def get(self, s):
		if s in self.default_static_src:
			return self.replace(self.default_static_tgt[self.default_static_src.index(s)])

		feed_dict = {
			self.infer_model.src_placeholder: [self.bpe.apply_bpe_sentence(self.vocab.tokenizer(s))],
			self.infer_model.batch_size_placeholder: 1,
		}	
		self.infer_sess.run(self.infer_model.iterator.initializer, feed_dict=feed_dict)
		nmt_outputs, attention_summary = self.loaded_infer_model.decode(self.infer_sess)
		sentences = [get_sentence(a).decode("utf-8").strip() for a in nmt_outputs]
		scoring = Scoring(s, sentences)
		res = scoring.get_best_scores(2)
		return self.replace(res[randint(0, 1)])