# -*- coding: utf-8 -*-
'''
	LyaBot, Prepare data
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
import argparse

from preprocessing import Preprocessing
from bpe import BPE

def main(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-vocab', dest='no_vocab', action='store_false')
	parser.add_argument('--no-apply_bpe', dest='no_apply_bpe', action='store_false') # If only apply or learn use json  (bpe_joins.json)
	


	args = parser.parse_args()
	format_data(args)

def format_data(args):
	preprocessing = Preprocessing(['data.src', 'data.tgt', 'train_1.src', 'train_1.tgt'])
	if args.no_vocab:
		preprocessing.create_vocab()
		preprocessing.learn_bpe()

	if args.no_apply_bpe:
		preprocessing.apply_bpe()	

if __name__ == "__main__":
	main()