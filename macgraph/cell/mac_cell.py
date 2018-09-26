
import tensorflow as tf

from .read_cell import *
from .control_cell import *
from .output_cell import *
from ..util import *




class MACCell(tf.nn.rnn_cell.RNNCell):

	def __init__(self, args, features, question_state, question_tokens, vocab_embedding):
		self.args = args
		self.features = features
		self.question_state = question_state
		self.question_tokens = question_tokens
		self.vocab_embedding = vocab_embedding

		super().__init__(self)

	def get_taps(self):
		return {
			"finished":				1,
			"question_word_attn": 	self.args["control_heads"] * self.features["d_src_len"],
			"kb_node_attn": 		self.args["kb_node_width"] * self.args["embed_width"],
			"kb_node_word_attn": 	self.args["kb_node_width"],
		}



	def __call__(self, inputs, in_state):
		"""Run this RNN cell on inputs, starting from the given state.
		
		Args:
			inputs: `2-D` tensor with shape `[batch_size, input_size]`.
			state: if `self.state_size` is an integer, this should be a `2-D Tensor`
				with shape `[batch_size, self.state_size]`.	Otherwise, if
				`self.state_size` is a tuple of integers, this should be a tuple
				with shapes `[batch_size, s] for s in self.state_size`.
			scope: VariableScope for the created subgraph; defaults to class name.
		Returns:
			A pair containing:
			- Output: A `2-D` tensor with shape `[batch_size, self.output_size]`.
			- New state: Either a single `2-D` tensor, or a tuple of tensors matching
				the arity and shapes of `state`.
		"""


		with tf.variable_scope("mac_cell", reuse=tf.AUTO_REUSE):

			in_control_state = in_state

			empty_attn = tf.fill([self.features["d_batch_size"], self.features["d_src_len"], 1], 0.0)
			empty_query = tf.fill([self.features["d_batch_size"], self.features["d_src_len"]], 0.0)

			out_control_state, tap_question_attn = control_cell(self.args, self.features, 
				inputs, in_control_state, self.question_state, self.question_tokens)
		
			read, read_taps = read_cell(
				self.args, self.features, self.vocab_embedding, out_control_state, 
				self.question_tokens, self.question_state)
			
			output, finished = output_cell(self.args, self.features,
				self.question_state, read, out_control_state)	
		
			out_state = out_control_state
			
			# TODO: Move this tap manipulation upstream, 
			#	have generic taps dict returned from the fns,
			#	and make this just use get_taps to append the data
			out_data  = [output, 
				tf.cast(finished, tf.float32),
				tap_question_attn,
				tf.squeeze(read_taps.get("kb_node_attn", empty_attn), 2),
				read_taps.get("kb_node_word_attn", empty_query),
			]

			return out_data, out_state



	@property
	def state_size(self):
		"""
		Returns a size tuple
		"""
		return self.args["control_width"]
		

	@property
	def output_size(self):
		return [
			self.args["output_classes"], 
		] + list(self.get_taps().values())





