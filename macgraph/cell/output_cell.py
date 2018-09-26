
import tensorflow as tf

from ..args import ACTIVATION_FNS


def output_cell(args, features, in_question_state, in_read, in_control_state):
	with tf.name_scope("output_cell"):

		v = tf.concat([in_read, in_control_state], -1)
		
		for i in range(args["output_layers"]):
			v = tf.layers.dense(v, args["output_classes"])
			v = ACTIVATION_FNS[args["output_activation"]](v)

		v = tf.layers.dense(v, args["output_classes"])

		return v