
import tensorflow as tf

from ..util import dynamic_assert_shape
from .text_util import UNK_ID


def get_table_with_embedding(args, features, vocab_embedding, noun):
	
	# --------------------------------------------------------------------------
	# Constants and validations
	# --------------------------------------------------------------------------

	table = features[f"{noun}s"]
	table_len = features[f"{noun}s_len"]

	width = args[f"{noun}_width"]
	full_width = width * args["embed_width"]

	d_len = tf.shape(table)[1]
	assert table.shape[-1] == width

	# --------------------------------------------------------------------------
	# Embed graph tokens
	# --------------------------------------------------------------------------
	
	emb_kb = tf.nn.embedding_lookup(vocab_embedding, table)
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, width, args["embed_width"]])

	emb_kb = tf.reshape(emb_kb, [-1, d_len, full_width])
	emb_kb = dynamic_assert_shape(emb_kb, 
		[features["d_batch_size"], d_len, full_width])

	return emb_kb, full_width, table_len

	