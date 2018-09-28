
import tensorflow as tf
import numpy as np
import yaml

from .cell import *
from .util import *
from .hooks import *
from .input import *
from .encoder import *

def model_fn(features, labels, mode, params):

	# --------------------------------------------------------------------------
	# Setup input
	# --------------------------------------------------------------------------

	args = params

	# EstimatorSpec slots
	loss = None
	train_op = None
	eval_metric_ops = None
	predictions = None
	eval_hooks = None

	vocab = Vocab.load(args)

	# --------------------------------------------------------------------------
	# Shared variables
	# --------------------------------------------------------------------------

	vocab_embedding = tf.get_variable(
		"vocab_embedding",
		[args["vocab_size"], args["embed_width"]],
		tf.float32)

	if args["use_summary_image"]:
		tf.summary.image("vocab_embedding", tf.reshape(vocab_embedding,
			[-1, args["vocab_size"], args["embed_width"], 1]))

	# --------------------------------------------------------------------------
	# Model for realz
	# --------------------------------------------------------------------------

	# Encode the input via biLSTM
	question_tokens, question_state = encode_input(args, features, vocab_embedding)

	# The control state is focusing on one of the input tokens
	out_control_state, tap_question_attn = control_cell(args, features, question_state, question_tokens)

	# The read cell pulls out the relevant node property from the graph
	read, read_taps = read_cell(
		args, features, vocab_embedding, out_control_state, 
		question_tokens)
	
	# The output cell transforms that property for output
	logits = output_cell(args, features,
		question_state, read, out_control_state)	

	# For visualisation of what attention is doing (try running predict.py)
	taps = {
		"question_word_attn": tap_question_attn,
		"kb_node_attn": 	  read_taps["kb_node_attn"],
		"kb_node_word_attn":  read_taps["kb_node_word_attn"],
	}

	# --------------------------------------------------------------------------
	# Calc loss
	# --------------------------------------------------------------------------

	if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
		crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		loss = tf.reduce_sum(crossent) / tf.to_float(features["d_batch_size"])

	# --------------------------------------------------------------------------
	# Optimize
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.TRAIN:
		global_step = tf.train.get_global_step()
		optimizer = tf.train.AdamOptimizer(args["learning_rate"])
		train_op, gradients = minimize_clipped(optimizer, loss, args["max_gradient_norm"])

		# Visualisation of training dynamics
		var = tf.trainable_variables()
		gradients = tf.gradients(loss, var)
		norms = [tf.norm(i, 2) for i in gradients if i is not None]
		tf.summary.scalar("grad_norm/max", tf.reduce_max(norms), family="hyperparam")
		tf.summary.scalar("grad_norm/avg", tf.reduce_mean(norms), family="hyperparam")


	# --------------------------------------------------------------------------
	# Predictions
	# --------------------------------------------------------------------------

	if mode in [tf.estimator.ModeKeys.PREDICT, tf.estimator.ModeKeys.EVAL]:

		predicted_labels = tf.argmax(tf.nn.softmax(logits), axis=-1)

		predictions = {
			"predicted_label": predicted_labels,
			"actual_label": features["label"],
		}

		# For diagnostic visualisation
		predictions.update(features)
		predictions.update(taps)

		# Fake features do not have batch, must be removed
		del predictions["d_batch_size"]
		del predictions["d_src_len"]

	# --------------------------------------------------------------------------
	# Eval metrics
	# --------------------------------------------------------------------------

	if mode == tf.estimator.ModeKeys.EVAL:

		eval_metric_ops = {
			"accuracy": tf.metrics.accuracy(labels=labels, predictions=predicted_labels),
		}

		# Add per class and per question type accuracy metrics

		try:
			with tf.gfile.GFile(args["question_types_path"]) as file:
				doc = yaml.load(file)
				for type_string in doc.keys():
					if args["type_string_prefix"] is None or type_string.startswith(args["type_string_prefix"]):
						eval_metric_ops["type_accuracy_"+type_string] = tf.metrics.accuracy(
							labels=labels, 
							predictions=predicted_labels, 
							weights=tf.equal(features["type_string"], type_string))


			with tf.gfile.GFile(args["answer_classes_path"]) as file:
				doc = yaml.load(file)
				for answer_class in doc.keys():
					e = vocab.lookup(pretokenize_json(answer_class))
					weights = tf.equal(labels, tf.cast(e, tf.int64))
					eval_metric_ops["class_accuracy_"+str(answer_class)] = tf.metrics.accuracy(
						labels=labels, 
						predictions=predicted_labels, 
						weights=weights)

		except tf.errors.NotFoundError as err:
			print(err)
			pass
		except Exception as err:
			print(err)
			pass


		eval_hooks = [FloydHubMetricHook(eval_metric_ops)]

	return tf.estimator.EstimatorSpec(mode,
		loss=loss,
		train_op=train_op,
		predictions=predictions,
		eval_metric_ops=eval_metric_ops,
		export_outputs=None,
		training_chief_hooks=None,
		training_hooks=None,
		scaffold=None,
		evaluation_hooks=eval_hooks,
		prediction_hooks=None
	)
