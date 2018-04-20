# -*- coding: utf-8 -*-
'''
	LyaBot, Data Formatter
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
	Inspired by https://github.com/pender/chatbot-rnn/tree/master/reddit-parse (https://github.com/pender/)
'''

import bz2
import ujson as json
import regex as re
import os
import sys
import bz2
import glob
import argparse
import lzma

from file_utils import xz_read_lines
from multiprocessing import Pool, Manager
from settings import settings


def main(): 
	parser = argparse.ArgumentParser()
	parser.add_argument('--path_reddit', type=str, default=settings.path_reddit, help='reddit data directory')
	parser.add_argument('--path_data', type=str, default=settings.path_data, help='directory to save the file')
	parser.add_argument('--min_score', type=int, default=7, help='minimum reddit score')
	parser.add_argument('--processes', type=int, default=10, help='processes')
	parser.add_argument('--cache_size', type=int, default=int(1e6), help='max comments in cache (used for parenting)')
	parser.add_argument('--batch_size', type=int, default=int(1e5), help='number of lines to read per iteration')
	parser.add_argument('--lines_per_file', type=int, default=int(1e6), help='maximum numbe of lines per generated file')
	args = parser.parse_args()
	parse(args)

class RedditComment(object):
	''' TLDW '''

	regex = {
		'sp' : re.compile('[ \t\r]+'),
		'ca' : re.compile('\^'),
		'url_1'  : re.compile(r'http[s]?:(//)?(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.IGNORECASE),
		'url_2'  : re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?/[-A-Za-z0-9_\\$\\.\\+\\!\\*\\(\\),;:@&:\\?/~\\#\\%]*[^]'\\.}>\\),\\\"]"),
		'url_3'  : re.compile("([0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}|(((news|telnet|nttp|file|http|ftp|https)://)|(www|ftp)[-A-Za-z0-9]*\\.)[-A-Za-z0-9\\.]+)(:[0-9]*)?"),
	}

	def __init__(self, json_data):
		self.body = RedditComment.format(json_data['body'])
		self.score = json_data['score']
		self.parent_id = json_data['parent_id']
		self.child_ids = []
		self.id = json_data['id']
		self.subreddit = json_data['subreddit']
		self.json_data = json_data
		self.i = 0
		# From what I understand, only t1 indicate a comment.
		if self.parent_id.startswith('t1_'):
			self.parent_id = self.parent_id
		else:
			self.parent_id = None

	@staticmethod
	def has_url(s):

		if RedditComment.regex['url_1'].search(s):
			return True

		if RedditComment.regex['url_2'].search(s):
			return True

		if RedditComment.regex['url_3'].search(s):
			return True

		return False

	@staticmethod
	def format(body):
		#Preprocess the comment text.
		body = re.sub(RedditComment.regex['sp'], ' ', body) # Replace runs of whitespace with a single space.
		body = re.sub(RedditComment.regex['ca'], '', body) # Strip out carets.
		return body

	@staticmethod
	def validate(comment, substring_blacklist, subreddit_whitelist, min_score):
		if comment.id is None:
			return False

		if comment.body is None:
			return False

		if comment.score is None:
			return False

		len_body = len(comment.body)
		len_body_words = len(comment.body.split(' '))
		if len_body < 4 or len_body > 300: 
			return False

		if len_body_words < 1 or len_body_words > 30:
			return False

		if comment.subreddit.lower() not in subreddit_whitelist:
			return False

		if substring_blacklist.search(comment.body) != None:
			return False

		if not comment.id.startswith('t1_'):
			comment.id = 't1_'+comment.id

		if RedditComment.has_url(comment.body):
			return False

		return comment


class RedditParse(object):
	''' It's a fucking prentious name for a class '''
	def __init__(self, comments, min_score):
		self.subreddit_whitelist = [  
			'marvel', 'marvelstudios', 'defenders', 'jessicajones',
			'webdev', 'gamedev', 'programming', 'indiedev', 
			'psychology', 'psych', 'psychologystudents', 'clinicalpsych',
			'philosophy', 'askphilosophy', 'philosophyofscience',
			'atheism', 'DebateAnAtheist', 'atheist', 'atheists',
			'egalitarianism', 'egalitarian',
			'askreddit', 'netflix', 'needafriend'
			'magic', 'cardmagic',
			'python', 'learnpython', 'tenserflow', 'unity3d', 'unity2d',
			'tesla', 'physics', 'asksciencediscussion', 'science',
			'geek', 'geekygirls', 'opensource', 'comicbooks', 'technology',
			'the100', 'gameofthrones', 'housemd', 'sense8', 'suits',
			'hearthstone', 'pathofexile', 'factorio', 'onepiece', 'fairytail',
			'funny', 'todayilearned', 'gaming', 'movies'
		]

		self.substring_blacklist = [
				# KILLING ALL THE MEMES.
				# ALL YOUR MEME ARE BELONG TO US.
				'\n', '\[', ' r/',' u/', '/r/', '/u/',
				'reddit',  'upvote', 'OOC:', 'upvoted', 'downvoted',
				'downvote', 'upvoting', 'downvoting',  
				'username checks out', 'username checksout',
				'dozens of us', 's treason then'
			]

		self.substring_blacklist = re.compile('|'.join(self.substring_blacklist), re.IGNORECASE)

		self.min_score = min_score
		self.comments = comments
		self._comments = {}	

	def parent_comments(self):
		self._comments = dict(self.comments)
		i = 0
		for comment in self._comments.values():
			if comment.parent_id != None and comment.parent_id in self._comments.keys():
				#if self._comments[comment.parent_id].child_id is None:
				if comment.score >= self.min_score and comment.id not in self._comments[comment.parent_id].child_ids:
					self._comments[comment.parent_id].child_ids.append(comment.id)
					i += 1
				#else:
				#	prev_child = self._comments[self._comments[comment.parent_id].child_id]
				#	if self._comments[comment.parent_id].score > prev_child.score:
				#		self._comments[comment.parent_id].child_id = comment.id
		return i


	def save(self, fr, to):
		i = 0

		for comment in self._comments.values():
			if len(comment.child_ids) > 0:
				for child_id in comment.child_ids:
					fr.write(comment.body + '\n')
					to.write(self._comments[child_id].body + '\n')
					i+= 1
				
		self._comments = {}

		return i

	def read_comment(self, line):
		if (line[-1] == '}' or line[-2] == '}' and len(line) <= 700):
			json_comment = json.loads(line)
			comment = RedditComment.validate(RedditComment(json_comment), self.substring_blacklist, self.subreddit_whitelist, self.min_score)
			if comment != False:
				self.comments[comment.id] = comment
				return 1
			return 0
		return 0

def parse(args):
	files = glob.glob('{}/RC_*.xz'.format(args.path_reddit))
	comments = Manager().dict()
	reddit_parse = RedditParse(comments, args.min_score)
	output_index = 0
	output_from = open(os.path.join(args.path_data, 'train_{}.src'.format(output_index)), mode="wt", encoding='utf-8')
	output_to = open(os.path.join(args.path_data, 'train_{}.tgt'.format(output_index)), mode="wt", encoding='utf-8')
	total = 0

	count_read = 0
	count_processed = 0
	count_saved = 0
	cycles = 1
	count_read_total = 0
	count_saved_total = 0

	for file in files:
		print ('*** STARTING FILE {} : GOOD LUCK! ^_^'.format(file))

		with Pool(processes=args.processes) as p:
			with lzma.open(file, "rt") as raw_data:
				for lines in xz_read_lines(raw_data, args.batch_size):

					count_read += sum(p.map(reddit_parse.read_comment, lines))
					count_read_total += count_read

					print ('\t** Total accepted: {} | Cycle {} | {} comments accepted over {}'.format(count_read_total, cycles, count_read, len(lines)),  end='\r', flush=True) 
					count_read = 0

					cycles += 1
					if count_read_total >= args.cache_size:
						print('\n')
						print ('\t** Starting to process {} comments ...'.format(len(reddit_parse.comments)))
						count_processed = reddit_parse.parent_comments()
						print ('\t** Processed {} comments, found {} with parents'.format(count_read_total, count_processed),  end='\r', flush=True)

						count_saved = reddit_parse.save(output_from, output_to)
						count_saved_total += count_saved
						total += count_saved
						print('\n')
						print ('\t** Saved {} new comments | TOTAL: {} comments'.format(count_saved, total),  end='\r', flush=True)
						count_read_total = 0
						count_processed = 0
						count_saved = 0
						
						if count_saved_total > args.lines_per_file:
							output_index += 1
							count_saved_total = 0
							output_from.close()
							output_to.close()
							output_from = open(os.path.join(args.path_data, 'train_{}.src'.format(output_index)), mode="wt", encoding='utf-8')
							output_to = open(os.path.join(args.path_data, 'train_{}.tgt'.format(output_index)), mode="wt", encoding='utf-8')	

						comments = Manager().dict()
						reddit_parse = RedditParse(comments, args.min_score)
						count_read = 0
						count_processed = 0
						count_saved = 0
						count_read_total = 0


if __name__ == "__main__":
	main()