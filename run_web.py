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

from flask import Flask, request, jsonify, make_response
from settings import settings
from model_utils import create_or_load_model, create_infer_model
from misc_utils import safe_exp, format_spm_text, load_data, get_sentence, run_infer_sample
from preprocessing import Preprocessing
from model import Model


def main():
	app = Flask(__name__)
	data = Preprocessing(None)


	infer_model = create_infer_model(Model)
	infer_sess = tf.Session( graph=infer_model.graph,  config=tf.ConfigProto(device_count={'GPU': 0}))
	with infer_model.graph.as_default():
		loaded_infer_model, global_step = create_or_load_model(infer_model.model, infer_sess, "infer")


	@app.route('/reply', methods=['POST', 'GET'])
	def reply():
		question = request.args.get('question')

		feed_dict = {
			infer_model.src_placeholder: [data.apply_bpe_sentence(data.tokenizer(question))],
			infer_model.batch_size_placeholder: 1,
		}	

		infer_sess.run(infer_model.iterator.initializer, feed_dict=feed_dict)

		nmt_outputs, attention_summary = loaded_infer_model.decode(infer_sess)

		r = []

		for rr in nmt_outputs:
			r.append(get_sentence(rr).decode("utf-8"))

		resp = make_response(jsonify({'replies': r}))
		resp.headers['Access-Control-Allow-Origin'] = '*'
		return resp


	@app.route('/', methods=['GET'])
	def chat():
		return  'lol'

	print('*** Web service started.')

	app.run(port=5001, host='0.0.0.0')


	return 'chat page'

if __name__ == "__main__":
	main()
