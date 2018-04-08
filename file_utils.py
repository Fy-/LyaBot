# -*- coding: utf-8 -*-
'''
	LyaBot, File utils
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

import regex as re
from itertools import takewhile, repeat
import os
import math

_LINES_IN_FILE = {} # used by read_lines(), cache of lines per files if we open the same fat file twice (or more).
_FILE_BATCH_SIZE = {}

def divisor_generator(n):
	''' Math sux '''
	large_divisors = []
	for i in range(1, int(math.sqrt(n) + 1)):
		if n % i == 0:
			yield i
			if i*i != n:
				large_divisors.append(n / i)

	for divisor in reversed(large_divisors):
		yield divisor

def closest_divisor(n, t):
	''' See above '''
	r = 0
	d = math.inf 
	for i in list(divisor_generator(n)):
		if d > math.fabs(t-i):
			r = i
			d = math.fabs(t-r)

	return r

def lines_in_file(file_path):
	''' "Fast" way to count lines in a file '''
	f = open(file_path, 'rb')
	buf_gen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
	return sum( buf.count(b'\n') for buf in buf_gen )

def read_lines(file, file_path, batch_size):
	''' Read (bacth_size) lines from {file} at {file_path} an return zip(lines) 
		Can be slow depending on Math Jesus. 
	'''
	_LINES_IN_FILE[file_path] = lines_in_file(file_path)
	_FILE_BATCH_SIZE[file_path] = int(closest_divisor(_LINES_IN_FILE[file_path], batch_size))

	l = [iter(file)] * _FILE_BATCH_SIZE[file_path]

	return zip(*l)

def read_until_for(file, fr, to):
	''' The title says everything imho '''
	ret = []
	i = 0
	for l in file:
		if i > fr:
			ret.append(l)
		if i >= to:
			return ret
		i += 1

def xz_read_lines(file, batch_size):
	''' Read (bacth_size) lines from as xz file and return the zip(lines)
		We will miss one batch, but it's ok because it's to slow if we don't. 
		This way, it's fast.

		It's name xz_read_lines for sentimental reasons.
	'''
	l = [iter(file)] * batch_size
	return zip(*l)

def write_lines(file, lines):
	''' Used only for multithreading writting. (or not right now)'''
	file.write(lines)
	return len(lines)

def load_data_readlines(inference_input_file):
	with open(inference_input_file, 'r', encoding='utf-8') as f:
		inference_data = f.read().splitlines()
	return inference_data
