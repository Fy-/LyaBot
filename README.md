# LyaBot - A "smart" chatbot
First, this is not like a real project or anything it was more like a learning experiencce for me at that's why I decided to make this repo public. The code is based and inspired by multiple greats projects:
- https://github.com/daniel-kukiela/nmt-chatbot (a cool chatbot, please check it out)
- https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/assignments/chatbot (gread tutorial for previous version of tensorflow)
- https://gist.github.com/higepon/eb81ba0f6663a57ff1908442ce753084 (minimum seq2seq implementation with tensorflow)
- https://github.com/bshao001/ChatLearner/ (A chatbot)
- And of course tensorflow/nmt:  https://github.com/tensorflow/nmt

# Model in short
```python
''' embeddings (always join vocab) '''
embedding_encoder = tf.get_variable("embedding_share", [vocab_size, num_units])
embedding_decoder = embedding_encoder

''' encoder '''
encoder_cell = create_encoder_cell()
encoder_emb_inputs = tf.nn.embedding_lookup(embedding_encoder, encoder_inputs, ...)

encoder_outputs, encoder_state = build_bidirectional_rnn(encoder_cell, ...)

''' output layer '''
layers_core.Dense(vocab_size, ...)

''' decoder '''
decoder_cell = create_decoder_cell() # use attention (LuongAttention)
decoder_emb_inputs = tf.nn.embedding_lookup(embedding_decoder, target_inputs)

helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inputs, ...)
decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, ...)
outputs, _final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, ...)
sample_id = outputs.sample_id
logits = output_layer(outputs.rnn_output)

''' Loss '''
crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=decoder_outputs, logits=logits)
train_loss = (tf.reduce_sum(crossent * target_weights) / batch_size)

''' gradient & optimization '''
params = tf.trainable_variables()
gradients = tf.gradients(train_loss, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
update_step = optimizer.apply_gradients(zip(clipped_gradients, params))

''' beam search '''
encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=settings.beam_width)
inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
	cell=decoder_cell,
	embedding=embedding_decoder,
	start_tokens=start_tokens,
	end_token=end_token,
	initial_state=encoder_state,
	beam_width=settings.beam_width,
	output_layer=output_layer,
	length_penalty_weight=1.0
)
outputs, _final_state, _final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(...)
sample_id = outputs.predicted_ids
```

 
# Datasets
I use reddit for my dataset and cornell movie dialogs and some customs files (check _data_static/_default.(src|tgt)) (http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html).
Concerning reddit you can download the datasets at http://files.pushshift.io/reddit/comments/ (please consider making a donation).
