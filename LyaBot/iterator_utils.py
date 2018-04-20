# -*- coding: utf-8 -*-
'''
	LyaBot, Iterator utils
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
	This file is based and inspired by https://github.com/tensorflow/nmt
'''
import os
import tensorflow as tf
from .settings import settings

class DataIterator(object):
	@staticmethod
	def get_infer_iterator(src_dataset, src_vocab_table, batch_size, src_max_len=None):
		src_eos_id = tf.cast(src_vocab_table.lookup(tf.constant(settings.eos)), tf.int32)
		src_dataset = src_dataset.map(lambda src: tf.string_split([src]).values)


		# Convert the word strings to ids
		src_dataset = src_dataset.map(lambda src: tf.cast(src_vocab_table.lookup(src), tf.int32))
		# Add in the word counts.
		src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

		def batching_func(x):
			return x.padded_batch(
				batch_size,
				# The entry is the source line rows;
				# this has unknown-length vectors.  The last entry is
				# the source row size; this is a scalar.
				padded_shapes=(
						tf.TensorShape([None]),  # src
						tf.TensorShape([])  # src_len
					),
				# Pad the source sequences with eos tokens.
				# (Though notice we don't generally need to do this since
				# later on we will be masking out calculations past the true sequence.
				padding_values=(
						src_eos_id,  # src
						0 # src_len -- unused
					)
				)
		batched_dataset = batching_func(src_dataset)
		batched_iter = batched_dataset.make_initializable_iterator()
		(src_ids, src_seq_len) = batched_iter.get_next()

		return type('',(object,), {
				'initializer' : batched_iter.initializer,
				'source' : src_ids,
				'target_input' : None,
				'target_output' : None,
				'source_sequence_length' : src_seq_len,
				'target_sequence_length' : None
			})

	@staticmethod
	def get_iterator(vocab_table, file=None, src_dataset=None, tgt_dataset=None, skip_count=None, reshuffle_each_iteration=True):
		''' Tensorflow is cool, even for reading and parsing files '''
		

		output_buffer_size = settings.batch_size * 1000

		num_buckets = settings.num_buckets

		batch_size = settings.batch_size

		if file != None:
			src_path = os.path.join(settings.data_formated, '{}{}'.format(file, '.bpe.src'))
			tgt_path = os.path.join(settings.data_formated, '{}{}'.format(file, '.bpe.tgt'))
			src_dataset = tf.data.TextLineDataset([src_path])
			tgt_dataset = tf.data.TextLineDataset([tgt_path])

		src_eos_id = tf.cast(vocab_table.lookup(tf.constant(settings.eos)), tf.int32)
		tgt_sos_id = tf.cast(vocab_table.lookup(tf.constant(settings.sos)), tf.int32)
		tgt_eos_id = tf.cast(vocab_table.lookup(tf.constant(settings.eos)), tf.int32)

		dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

		if skip_count is not None:
			dataset = dataset.skip(skip_count)

		dataset = dataset.shuffle(output_buffer_size, None, reshuffle_each_iteration)

		# Split 
		dataset = dataset.map(
			lambda src, tgt: (
				tf.string_split([src]).values, tf.string_split([tgt]).values
			),
			num_parallel_calls=10).prefetch(output_buffer_size)

		# Filter zero length input sequences. 
		dataset = dataset.filter(lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

		# Convert the word strings to ids. if not default int
		dataset = dataset.map(
			lambda src, tgt: (
				tf.cast(vocab_table.lookup(src), tf.int32),
				tf.cast(vocab_table.lookup(tgt), tf.int32)
			),
			num_parallel_calls=10).prefetch(output_buffer_size)

		# Create a tgt_input prefixed with <sos> and a tgt_output suffixed with <eos>.
		dataset = dataset.map(
			lambda src, tgt: (src,
				tf.concat(([tgt_sos_id], tgt), 0),
				tf.concat((tgt, [tgt_eos_id]), 0)
			),
			num_parallel_calls=10).prefetch(output_buffer_size)

		# Add in sequence lengths.
		dataset = dataset.map(
			lambda src, tgt_in, tgt_out: (
				src, 
				tgt_in, 
				tgt_out, 
				tf.size(src), 
				tf.size(tgt_in)
			),
			num_parallel_calls=10).prefetch(output_buffer_size)

		def batching_func(x):
			return x.padded_batch(
				batch_size,
				# The first three entries are the source and target line rows;
				# these have unknown-length vectors.  The last two entries are
				# the source and target row sizes; these are scalars.
				padded_shapes=(
					tf.TensorShape([None]),  # src
					tf.TensorShape([None]),  # tgt_input
					tf.TensorShape([None]),  # tgt_output
					tf.TensorShape([]),  # src_len
					tf.TensorShape([])),  # tgt_len
				# Pad the source and target sequences with eos tokens.
				# (Though notice we don't generally need to do this since
				# later on we will be masking out calculations past the true sequence.
				padding_values=(
					src_eos_id,  # src
					tgt_eos_id,  # tgt_input
					tgt_eos_id,  # tgt_output
					0,  # src_len -- unused
					0) # tgt_len -- 
				)  

		if num_buckets > 1:

			def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
				# Calculate bucket_width by maximum source sequence length.
				# Pairs with length [0, bucket_width) go to bucket 0, length
				# [bucket_width, 2 * bucket_width) go to bucket 1, etc.  Pairs with length
				# over ((num_bucket-1) * bucket_width) words all go into the last bucket.
				bucket_width = (50 + num_buckets - 1) // num_buckets

				# Bucket sentence pairs by the length of their source sentence and target
				# sentence.
				bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
				return tf.to_int64(tf.minimum(num_buckets, bucket_id))

			def reduce_func(unused_key, windowed_data):
				return batching_func(windowed_data)

			batched_dataset = dataset.apply(
					tf.contrib.data.group_by_window(
							key_func=key_func, reduce_func=reduce_func, window_size=batch_size))
		else:
			batched_dataset = batching_func(dataset)

		batched_iter = batched_dataset.make_initializable_iterator()
		(src_ids, tgt_input_ids, tgt_output_ids, src_seq_len, tgt_seq_len) = (batched_iter.get_next())

		return type('',(object,), {
			'initializer' : batched_iter.initializer,
			'source' : src_ids,
			'target_input' : tgt_input_ids,
			'target_output' : tgt_output_ids,
			'source_sequence_length' : src_seq_len,
			'target_sequence_length' : tgt_seq_len
		})
