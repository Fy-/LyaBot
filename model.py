# -*- coding: utf-8 -*-
'''
	LyaBot, Data Formatter
	~~~~~~~~~~~~~~~~~~~~~~
	:copyright: (c) 2018 by Gasquez Florian
	:license: MIT, see LICENSE for more details.
	
	This file is based and inspired by https://github.com/tensorflow/nmt
'''

import tensorflow as tf
from tensorflow.python.layers import core as layers_core
from settings import settings
from model_utils import create_rnn_cell, gradient_clip

class Model(object):
	def __init__(self, mode, iterator, vocab_table, reverse_vocab_table=None):
		self.mode = mode
		self.iterator = iterator
		self.vocab_table = vocab_table
		self.reverse_vocab_table = reverse_vocab_table


		initializer = tf.random_uniform_initializer(minval=-settings.init_weight, maxval=settings.init_weight)
		tf.get_variable_scope().set_initializer(initializer)

		# Embeddings
		self._embbedings()
		self.batch_size = tf.size(self.iterator.source_sequence_length)

		# Projection
		with tf.variable_scope("build_network"):
			with tf.variable_scope("decoder/output_projection"):
				self.output_layer = layers_core.Dense(settings.vocab_size, use_bias=False, name="output_projection")

		# Train graph
		res = self.build_graph()

		if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
			self.train_loss = res[1]
			self.word_count = tf.reduce_sum(self.iterator.source_sequence_length) + tf.reduce_sum(self.iterator.target_sequence_length)
			self.predict_count = tf.reduce_sum(self.iterator.target_sequence_length)
		elif self.mode == tf.contrib.learn.ModeKeys.INFER:
			self.infer_logits, _, self.final_context_state, self.sample_id = res
			self.sample_words = reverse_vocab_table.lookup(tf.to_int64(self.sample_id))

		self.global_step = tf.Variable(0, trainable=False)
		self.global_epoch = tf.Variable(0, trainable=False)

		params = tf.trainable_variables()

		if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
			self.learning_rate = tf.constant(settings.learning_rate)
			self.learning_rate = self._get_learning_rate_decay()
			
			# Optimizer
			opt = tf.train.AdamOptimizer(self.learning_rate)

			# Gradients
			gradients = tf.gradients(self.train_loss, params, colocate_gradients_with_ops=True)
			clipped_gradients, gradient_norm_summary, gradient_norm = gradient_clip(gradients, max_gradient_norm=settings.max_gradient_norm)

			self.gradient_norm = gradient_norm
			self.update = opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

			# Summary
			self.train_summary = tf.summary.merge([
				tf.summary.scalar("lr", self.learning_rate),
				tf.summary.scalar("train_loss", self.train_loss),
			] + gradient_norm_summary)

		elif self.mode == tf.contrib.learn.ModeKeys.INFER:
			self.infer_summary = tf.no_op()

		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=settings.num_keep_ckpts)

		if (self.mode == tf.contrib.learn.ModeKeys.TRAIN):
			print('*** Trainable variables')
			for param in params:
				print("\t *** %s, %s, %s" % (param.name, str(param.get_shape()), param.op.device))


	def decode(self, sess):
		_, infer_summary, _, sample_words = self.infer(sess)
		sample_words = sample_words.transpose()
		return sample_words, infer_summary

	def train(self, sess):
		''' You hope ... '''
		assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
		return sess.run(
			[
				self.update,
				self.train_loss,
				self.predict_count,
				self.train_summary,
				self.global_step,
				self.word_count,
				self.batch_size, 
				self.gradient_norm,
				self.learning_rate
			]
		)

	def infer(self, sess):
		assert self.mode == tf.contrib.learn.ModeKeys.INFER
		return sess.run(
			[
				self.infer_logits, 
				self.infer_summary, 
				self.sample_id, 
				self.sample_words
			]
		)

	def _embbedings(self):
		with tf.variable_scope("embeddings"):
			self.embedding_encoder = tf.get_variable("embedding_share", [settings.vocab_size, settings.num_units])
			self.embedding_decoder = self.embedding_encoder

	def _max_iters(self, source_sequence_length):
		''' Max decoding steps at infer time '''
		decoding_length_factor = 2.0
		max_encoder_length = tf.reduce_max(source_sequence_length)
		maximum_iterations = tf.to_int32(tf.round(tf.to_float(max_encoder_length) * decoding_length_factor))
		return maximum_iterations

	def _decoder(self, encoder_outputs, encoder_state):
		tgt_sos_id = tf.cast(self.vocab_table.lookup(tf.constant(settings.sos)), tf.int32)
		tgt_eos_id = tf.cast(self.vocab_table.lookup(tf.constant(settings.eos)), tf.int32)

		iterator = self.iterator
		maximum_iterations = self._max_iters(iterator.source_sequence_length) 

		with tf.variable_scope("decoder") as scope:
			decoder_cell, decoder_initial_state = self._decoder_cell(encoder_outputs, encoder_state)
			if self.mode != tf.contrib.learn.ModeKeys.INFER:
				target_inputs = iterator.target_input
				target_inputs = tf.transpose(target_inputs)
				decoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_decoder, target_inputs)
	
				# Helper
				helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, iterator.target_sequence_length, time_major=True)

				# Decoder
				decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, decoder_initial_state)
				outputs, _final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, output_time_major=True, swap_memory=True, scope=scope)
				sample_id = outputs.sample_id
				logits = self.output_layer(outputs.rnn_output)

			elif self.mode == tf.contrib.learn.ModeKeys.INFER:
				beam_width = settings.beam_width	


				start_tokens = tf.fill([self.batch_size], tgt_sos_id)
				end_token = tgt_eos_id


				# Define a beam-search decoder
				inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
					cell=decoder_cell,
					embedding=self.embedding_decoder,
					start_tokens=start_tokens,
					end_token=end_token,
					initial_state=decoder_initial_state,
					beam_width=beam_width,
					output_layer=self.output_layer,
					length_penalty_weight=1.0
				)

				# Dynamic decoding
				outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
						inference_decoder, 
						maximum_iterations=maximum_iterations, 
						output_time_major=True, 
						swap_memory=True,
						scope=scope
					)

				sample_id = outputs.predicted_ids
				logits = tf.no_op()

		return logits, sample_id, _final_state

	def _build_bidirectional_rnn(self, inputs, sequence_length, num_layers, dtype):
		fw_cell = self._encoder_cell(num_layers)
		bw_cell = self._encoder_cell(num_layers)
		bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
			fw_cell, bw_cell,
			inputs, sequence_length=sequence_length,
			time_major=True, swap_memory=True,
			dtype=dtype
		)
		return tf.concat(bi_outputs, -1), bi_state

	def _encoder(self):
		source = self.iterator.source
		source = tf.transpose(source) # Time major always true

		with tf.variable_scope("encoder") as scope:
			encoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_encoder, source)
			num_bi_layers = int(settings.num_layers / 2) #settings.num_layers
			encoder_outputs, bi_encoder_state = (
					self._build_bidirectional_rnn(
						inputs=encoder_emb_inputs,
						sequence_length=self.iterator.source_sequence_length,
						num_layers=num_bi_layers, dtype=scope.dtype
					)
				)

			if num_bi_layers == 1:
				encoder_state = bi_encoder_state
			else:
				encoder_state = []
				for layer_id in range(num_bi_layers):
					encoder_state.append(bi_encoder_state[0][layer_id])  # forward
					encoder_state.append(bi_encoder_state[1][layer_id])  # backward
				encoder_state = tuple(encoder_state)

			'''
			encoder_emb_inputs = tf.nn.embedding_lookup(self.embedding_encoder, source)

			encoder_cell = create_rnn_cell(
				num_units=settings.num_units,
				num_layers=settings.num_layers,
				forget_bias=False,
				dropout=settings.dropout,
				mode=self.mode
			)

			encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_emb_inputs, sequence_length=self.iterator.source_sequence_length, time_major=True, dtype=tf.float32)
			'''


		return encoder_outputs, encoder_state

	def _encoder_cell(self, num_layers):
		return create_rnn_cell(
				num_units=settings.num_units,
				num_layers=num_layers,
				forget_bias=0.1,
				dropout=settings.dropout,
				mode=self.mode
			)

	def _decoder_cell(self, encoder_outputs, encoder_state):
		'''
		cell = create_rnn_cell(
			num_units=settings.num_units,
			num_layers=settings.num_layers,
			forget_bias=False,
			dropout=settings.dropout,
			mode=self.mode
		)
		if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
			decoder_initial_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=settings.beam_width)
		else:
			decoder_initial_state = encoder_state

		return cell, decoder_initial_state
		'''

		'''Build a RNN cell with attention mechanism that can be used by decoder.'''

		dtype = tf.float32
		iterator = self.iterator
		beam_width = settings.beam_width
		memory = tf.transpose(encoder_outputs, [1, 0, 2])

		if self.mode == tf.contrib.learn.ModeKeys.INFER:
			memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
			source_sequence_length = tf.contrib.seq2seq.tile_batch(iterator.source_sequence_length, multiplier=beam_width)
			encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
			batch_size = self.batch_size * beam_width
		else:
			source_sequence_length = iterator.source_sequence_length
			batch_size = self.batch_size

		cell = create_rnn_cell(
			num_units=settings.num_units,
			num_layers=settings.num_layers,
			forget_bias=0.1,
			dropout=settings.dropout,
			mode=self.mode
		)

		attention_mechanism = tf.contrib.seq2seq.LuongAttention(
				settings.num_units, memory, 
				memory_sequence_length=source_sequence_length, 
				scale=True
			)
		cell = tf.contrib.seq2seq.AttentionWrapper(
				cell, attention_mechanism, attention_layer_size=settings.num_units, 
				alignment_history=False, output_attention=True, 
				name="attention"
			)

		#decoder_initial_state = cell.zero_state(batch_size, dtype)
		decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state)
		return cell, decoder_initial_state

	def get_max_time(self, tensor):
		return tensor.shape[0].value or tf.shape(tensor)[0]

	def _get_learning_rate_decay(self):
		'''Get learning rate decay.'''
		decay_factor = 0.5
		start_decay_step = int(settings.num_train_steps * 2 / 3)
		decay_times = 4
		remain_steps = settings.num_train_steps - start_decay_step
		decay_steps = int(remain_steps / decay_times)
		return tf.cond(
				self.global_step < start_decay_step,
				lambda: self.learning_rate,
				lambda: tf.train.exponential_decay(
					self.learning_rate,
					(self.global_step - start_decay_step),
					decay_steps, 
					decay_factor, 
					staircase=True
				), name="learning_rate_decay_cond"
			)

	def _get_loss(self, logits):
		target_output = self.iterator.target_output
		target_output = tf.transpose(target_output)
		max_time = self.get_max_time(target_output)
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_output, logits=logits)
		target_weights = tf.sequence_mask(self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
		target_weights = tf.transpose(target_weights)
		loss = tf.reduce_sum(crossent * target_weights) / tf.to_float(self.batch_size)
		return loss

	def build_graph(self):
		with tf.variable_scope("dynamic_seq2seq", dtype=tf.float32):
			# Encoder
			encoder_outputs, encoder_state = self._encoder()

			# Decoder
			logits, sample_id, final_context_state = self._decoder(encoder_outputs, encoder_state)

			if self.mode != tf.contrib.learn.ModeKeys.INFER:
				loss = self._get_loss(logits)
			else:
				loss = None

			return logits, loss, final_context_state, sample_id

