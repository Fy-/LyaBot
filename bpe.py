import regex as re
import ujson as json
import pickle
import os

from settings import settings
from collections import Counter, defaultdict
from vocab import Vocab

class BPE(object):
	def __init__(self, vocab = None):
		self.vocab = vocab
		self.vocab_obj = Vocab()
		self.joins = None
	def apply_bpe_sentence(self, sentence):
		sentence_cache = {}

		if self.joins is None:
			with open(settings.bpe_file, 'r', encoding='utf-8', buffering=131072) as bpe_file:
				self.joins = {tuple(json.loads(k)): v for k, v in json.load(bpe_file).items()}

		joins = self.joins
	
		# Split sentence by ' ▁'
		entities = self.vocab_obj.split(sentence)
		new_sentence = []

		# For every entity in sentence
		for entity in entities:

			# If entity exists in cache - used cached (computed earlier) result
			original_entity = entity
			if original_entity in sentence_cache:
				new_sentence.append(sentence_cache[original_entity])
				continue

			# Split entity into pieces (mostly chars)
			entity = entity.split()

			# Make pairs of neighboring pieces/chars
			pairs = []
			prev_char = entity[0]
			for char in entity[1:]:
				pairs.append((prev_char, char))
				prev_char = char

			# Single piece/char - nothing to join
			if not pairs:
				new_sentence.append(entity[0])
				continue

			# Make every possible join
			while True:

				# Joins fragment - includes only pairs that exists in current entity
				subjoins = {pair:joins[pair] for pair in pairs if pair in joins}
				
				# Find most common pair
				pair = min(subjoins, key=subjoins.get, default=())

				# If there's no one - entity is joined
				if not pair or pair not in pairs:
					break

				# prepare pieces/chars
				first, second = pair
				new_pair = first + second

				#print(pairs)

				# Replace every occurence of pair with a joied one
				while pair in pairs:

					# Find pair occurence
					index = pairs.index(pair)

					# Remove pair and update neighbour pairs with joined one
					if index > 0:
						pairs[index - 1] = (pairs[index - 1][0], new_pair)
					if index < len(pairs) - 1:
						pairs[index + 1] = (new_pair, pairs[index + 1][1])
					if len(pairs) == 1:
						pairs[0] = (new_pair, '')
					else:
						del pairs[index]

			# We are going to use first subword from pair to rebuild entity, so we need to add second subword of last entity as a new 'pair'
			# (AB, C), (C, DEF), (DEF, GHIJK) -> AB, C, DEF, GHIJK
			if pairs[-1][1]:
				pairs.append((pairs[-1][1], ''))
			nentity = ' '.join([first for (first, second) in pairs])
			new_sentence.append(nentity)
			sentence_cache[original_entity] = nentity

		# Return joined sentence
		return ' '.join(new_sentence)
		
	def learn_bpe(self, _vocab = None):
		if _vocab == None:
			_vocab = self.vocab

		stats = Counter() # Pair stats
		indices = defaultdict(lambda: defaultdict(int)) # Pairs indexes

		vocab = []
		train_vocab = Counter()
		for i, (entity, freq) in enumerate(_vocab.most_common()):
			# Split vocab token
			entity = tuple(entity.split())

			# Make pairs ("ABCD" -> (A, B), (B, C), (C, D)), stats, indexes and train vocab
			prev_char = entity[0]
			train_vocab[prev_char] += freq
			for char in entity[1:]:
				stats[prev_char, char] += freq
				indices[prev_char, char][i] += 1
				train_vocab[char] += freq
				prev_char = char
			vocab.append((entity, freq))


		print ('*** Learning BPE for vocab of {} tokens'.format(settings.vocab_wanted_size))
		_vocab = None
		self.vocab = None

		# List of joins per vocab
		joins = []

		# Partial stats speeds up learning process - optimization for 'max' above
		partial_stats = Counter(['', -1])
		partial_stats_min = -1
		update_partial_stats = True

		# Current number of vocab tokens
		train_vocab_len = prev_train_vocab_len = len(train_vocab)

		# Learn until vocab will contain desired number of tokens
		while train_vocab_len < settings.vocab_wanted_size:
		    print('\t *** BPE {}/{}'.format(prev_train_vocab_len, settings.vocab_wanted_size), end='\r', flush=True) 

		    clean_train_vocab = False

		    # Get most frequent pair
		    most_frequent, freq = partial_stats.most_common(1)[0]

		    # Update partial stats or frequency of most frequent pair is less than saved minimum for partial stats
		    if update_partial_stats or freq < partial_stats_min:
		        partial_stats_min = stats.most_common(500)[-1][1]
		        partial_stats = Counter()
		        for k, v in stats.most_common():
		            if v < partial_stats_min:
		                break
		            partial_stats[k] = v
		        update_partial_stats = False

		        # Get most frequent pair (again, proper one this time)
		        most_frequent, _ = partial_stats.most_common(1)[0]

		    # If frequency is lower than 2 - exit
		    if stats[most_frequent] < 2:
		        print('No pair has frequency greater than 1. Stopping earlier, your vocab file will include less tokens.\n')
		        break

		    # Replace pair "A B" with new entity "AB"

		    # Changes made
		    changes = []

		    # Replace regex
		    pattern = re.compile(r'(?<!\S)' + re.escape(' '.join(most_frequent)) + r'(?!\S)')

		    # Loop through indices
		    for j, freq in indices[most_frequent].items():

		        # Do not touch not existent pairs
		        if freq < 1:
		            continue

		        # Get entity and frequency
		        entity, freq = vocab[j]

		        # Replace "A B" with "AB" in entity
		        new_entity = pattern.sub(''.join(most_frequent), ' '.join(entity))
		        new_entity = tuple(new_entity.split())

		        # Update entity
		        vocab[j] = (new_entity, freq)

		        changes.append((j, new_entity, entity, freq))

		    # Update indices and pair stats
		    # Merged pair doesn't exist anymore
		    stats[most_frequent] = 0
		    partial_stats[most_frequent] = 0
		    indices[most_frequent] = defaultdict(int)

		    # Get entities and a new pair
		    first, second = most_frequent
		    new_pair = first + second

		    # Iterate through all changes
		    for j, entity, old_entity, freq in changes:

		        # Find all occurences of first pair entity
		        prev = -2
		        for i in iter([i for i, entity in enumerate(old_entity) if entity == first]):

		            # Do not touch second "B B" if "B B B"
		            if i == prev + 1:
		                continue

		            # Check if second pair entity follows first one
		            if i < len(old_entity) - 1 and old_entity[i + 1] == second:

		                # Reduce frequency of "A B" in "A B C D" where "B C" is a merged pair
		                if i:
		                    prev = old_entity[i - 1:i + 1]
		                    stats[prev] -= freq
		                    partial_stats[prev] = stats[prev]
		                    indices[prev][j] -= 1

		                # Reduce frequency of "C D" in "A B C D" where "B C" is a merged pair
		                if i < len(old_entity) - 2:

		                    # But do not touch "C B" if "A B C B C" as values will be adjusted with next occurence of "B C" pair
		                    if old_entity[i + 2] != first or i >= len(old_entity) - 3 or old_entity[i + 3] != second:
		                        next = old_entity[i + 1:i + 3]
		                        stats[next] -= freq
		                        partial_stats[next] = stats[next]
		                        indices[next][j] -= 1

		                prev = i

		                if train_vocab[first] <= freq or train_vocab[second] <= freq:
		                    clean_train_vocab = True
		                train_vocab[first] -= freq
		                train_vocab[second] -= freq

		        # Find all occurences of first pair entity
		        for i in [i for i, entity in enumerate(entity) if entity == new_pair]:

		            # Increase frequency of (new pair) "A BC" in "A BC D"
		            if i:
		                prev = entity[i - 1:i + 1]
		                stats[prev] += freq
		                #if stats[prev] >= partial_stats_min:
		                #    update_partial_stats = True
		                partial_stats[prev] = stats[prev]
		                indices[prev][j] += 1

		            # Increase frequency of (new pair) "BC D" in "A BC D", but do not touch if "A BC BC" as stats for "BC BC" will be adjusted win next occurence of "BC" pair
		            if i < len(entity) - 1 and entity[i + 1] != new_pair:
		                next = entity[i:i + 2]
		                stats[next] += freq
		                #if stats[next] >= partial_stats_min:
		                #    update_partial_stats = True
		                partial_stats[prev] = stats[prev]
		                indices[next][j] += 1

		            # Set frequency of a new pair
		            train_vocab[new_pair] += freq

		    # Current pair is merged - is not a pair anymore, so has frequency of 0
		    stats[most_frequent] = 0
		    partial_stats[most_frequent] = 0

		    # Remove (from training vocab) tokens with frequency of 0
		    if clean_train_vocab:
		        train_vocab= +train_vocab

		    # Calculate current number of train vocab entities
		    prev_train_vocab_len = train_vocab_len
		    train_vocab_len = len(train_vocab)
		    train_vocab_len_diff = train_vocab_len - prev_train_vocab_len


		    # Add new join pair
		    joins.append(most_frequent)

		# Save list of joins for train vocab
		joins = dict(reversed([(v, i) for i, v in enumerate(joins)]))

		with open(settings.bpe_file, 'w', encoding='utf-8', buffering=131072) as bpe_file:
			json.dump({json.dumps(k):v for k,v in joins.items()}, bpe_file)

		data_vocab = [entity for entity, _ in train_vocab.most_common()]
		with open(os.path.join(settings.data_formated, 'vocab.bpe.src'), 'w', encoding='utf-8')  as vocab_file:
			vocab_file.write(('\n'.join([settings.unk, settings.sos, settings.eos]) + '\n' + "\n".join(data_vocab)))

		self.joins = joins
		print('\n', end='\r', flush=True)
