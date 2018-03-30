# -*- coding: utf-8 -*-
"""
	LyaBot, Prepare data
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.

"""
import argparse

from preprocessing import Preprocessing


def main(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--no-vocab', dest='no_vocab', action='store_false')
	parser.add_argument('--no-apply_bpe', dest='no_apply_bpe', action='store_false') # If only apply or learn use json  (bpe_joins.json)


	args = parser.parse_args()
	format_data(args)

def format_data(args):
	preprocessing = Preprocessing() # Automaticly use all *.src/*tgr files in _data_raw.

	if not no_vocab:
		df.create_vocab()
		df.learn_bpe(vocab)

	if not no_apply_bpe:
		df.apply_bpe()	


if __name__ == "__main__":
	main()