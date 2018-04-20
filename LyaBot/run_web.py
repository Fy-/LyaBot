# -*- coding: utf-8 -*-
'''
	LyaBot, Web version
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


import tensorflow as tf
import time
from urllib.parse import unquote
from flask import Flask, request, jsonify, make_response
from .settings import settings
from .preprocessing import Preprocessing
from .model import Model
from .reply import Reply
from .model_utils import create_or_load_model, create_infer_model

app = Flask(__name__)

infer_model = create_infer_model(Model)
infer_sess = tf.Session( graph=infer_model.graph,  config=tf.ConfigProto(device_count={'GPU': 0}))

with infer_model.graph.as_default():
	loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, "infer")

replies = Reply(infer_model, infer_sess, loaded_infer_model)

@app.route('/reply', methods=['POST', 'GET'])
def reply():
	question = request.args.get('question')



	r = []
	question = unquote(question.strip())
	print(question)
	r.append(replies.get(question))

	resp = make_response(jsonify({'replies': r}))
	resp.headers['Access-Control-Allow-Origin'] = '*'
	return resp






if __name__ == "__main__":
	app.run(port=5001, host='0.0.0.0')

