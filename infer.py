# -*- coding: utf-8 -*-
'''
	LyaBot, Infer
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

from model import Model
from model_utils import create_or_load_model, create_infer_model
from eval_utils import format_spm_text, get_sentence, run_infer_sample
from bpe import BPE
from vocab import Vocab
from scoring import Scoring
from settings import settings
import tensorflow as tf
import time, os


if __name__ == "__main__":
	s = [
		'What is your name ?',
		'Good morning!',
		'Blue pill or red pill?',
		'What do you get if you multiply six by nine?',
		'What\'s up?',
		'What color is the Sky?',
		'What is your lastname?',
		'What\'s your name?',
		'What do you do when you\'re alone?',
		'Do you have cats?',
		'What do you think of God?',
		'Is there an afterlife?',
		'Does god exist?',
		'Why does god exist?',
		'What do you think of the existence of god?',
		'Is stealing a good thing? Why?',
		'Are you a robot?',
		'Are you human?',
		'Do you like Marvel?',
		'Who is Tony Stark?',
		'Do you watch game of thrones?',
		'Do you like Ned Stark?',
		'I love you, you\'re beautiful!',
		'I came in like a wrecking ball',
		'What\'s your favorite food?',
		'Are you a god?',
		'Do you like music?',
		'Fuck you!',
		'What are your favorite movies?',
		'Imagine all the people',
		'Linux is?',
		'Do you know bill gates?',
		'How old are you?',
		'Are you a feminist?'
	]


	bpe = BPE()
	vocab = Vocab()
	infer_model = create_infer_model(Model)

	infer_sess = tf.Session( graph=infer_model.graph,  config=tf.ConfigProto(device_count={'GPU': 0}))

	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, "infer")

	for ss in s:
		feed_dict = {
			infer_model.src_placeholder: [bpe.apply_bpe_sentence(vocab.tokenizer(ss))],
			infer_model.batch_size_placeholder: 1,
		}	
		infer_sess.run(infer_model.iterator.initializer, feed_dict=feed_dict)

		nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)


		print('\n*** Input  >>> "{}"'.format(
				ss
			)
		)

		sentences = [get_sentence(a).decode("utf-8").strip() for a in nmt_outputs]
		scoring = Scoring(ss, sentences)
		for k, v in scoring.get_best_scores(5).items():
			print ('*** LyaBot (score: {}) >>> "{}"'.format(v[0], v[1]))


