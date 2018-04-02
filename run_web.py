# -*- coding: utf-8 -*-
'''
	LyaBot, Web version
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.
	
'''


import tensorflow as tf

from flask import Flask, request, jsonify
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

		return jsonify({'replies': r})


	@app.route('/', methods=['GET'])
	def chat():
		return  'lol'

	print('*** Web service started.')

	app.run(port=5001, host='0.0.0.0')


	return 'chat page'

if __name__ == "__main__":
	main()