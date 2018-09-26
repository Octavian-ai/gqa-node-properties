
import tensorflow as tf

from ..util import *
from ..attention import *
from ..input import UNK_ID, get_table_with_embedding
from ..args import ACTIVATION_FNS


def read_from_table(args, features, in_signal, noun, table, width, keys_len=None):

	query = tf.layers.dense(in_signal, width)

	output, score_sm, total_raw_score = attention(table, query,
		key_width=width, 
		keys_len=keys_len,
	)

	output = dynamic_assert_shape(output, [features["d_batch_size"], width])
	return output, score_sm, table, total_raw_score


def read_from_table_with_embedding(args, features, vocab_embedding, in_signal, noun):
	"""Perform attention based read from table

	Will transform table into vocab embedding space
	
	@returns read_data
	"""

	with tf.name_scope(f"read_from_{noun}"):

		table, full_width, keys_len = get_table_with_embedding(args, features, vocab_embedding, noun)

		# --------------------------------------------------------------------------
		# Read
		# --------------------------------------------------------------------------

		return read_from_table(args, features, 
			in_signal, 
			noun,
			table, 
			width=full_width, 
			keys_len=keys_len)


def read_cell(args, features, vocab_embedding, in_control_state, in_question_tokens, in_question_state):
	"""
	A read cell

	@returns read_data

	"""


	with tf.name_scope("read_cell"):

		taps = {} # For visualisation of attention

		# --------------------------------------------------------------------------
		# Read data
		# --------------------------------------------------------------------------

		read, taps["kb_attn"], _, _ = read_from_table_with_embedding(
			args, 
			features, 
			vocab_embedding, 
			in_control_state, 
			noun="kb_node"
		)

		read_words = tf.reshape(read, [features["d_batch_size"], args["kb_node_width"], args["embed_width"]])	
		read, taps["kb_node_word_attn"] = attention_by_index(in_control_state, read_words)
		read = tf.concat([read, in_control_state], -1)
		read = tf.layers.dense(read, args["read_width"], activation=ACTIVATION_FNS[args["read_activation"]])
		
		# --------------------------------------------------------------------------
		# Prepare and shape results
		# --------------------------------------------------------------------------
	
		out_data = read

		# Residual skip connection
		# out_data = tf.concat([read, in_control_state], -1)
		
		# for i in range(args["read_layers"]):
		# 	out_data = tf.layers.dense(out_data, args["read_width"])
		# 	out_data = ACTIVATION_FNS[args["read_activation"]](out_data)
		
		return out_data, taps




