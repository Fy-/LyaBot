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

from settings import settings
from reply import Reply
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

	infer_model = create_infer_model(Model)

	infer_sess = tf.Session( graph=infer_model.graph,  config=tf.ConfigProto(device_count={'GPU': 0}))

	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, "infer")

	reply = Reply(infer_model, infer_sess, loaded_infer_model)

	for ss in s:
		print('\n*** Input  >>> "{}"'.format(
				ss
			)
		)
		print(reply.get(ss))

