# -*- coding: utf-8 -*-
"""
	LyaBot, Scoring
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.
"""
from misc_utils import safe_exp, format_spm_text

class Scoring(object):
	def __init__(self, outputs):
		self.outputs = outputs
		self.sentences = []