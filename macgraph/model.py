
import tensorflow as tf
import numpy as np
import yaml

from .cell import execute_reasoning
from .encoder import encode_input
from .cell import *
from .util import *
from .hooks import *
from .input import *

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

	# Run the cell once to get its output
	d_cell = MACCell(args, features, question_state, question_tokens, vocab_embedding)
	d_cell_initial = d_cell.zero_state(dtype=tf.float32, batch_size=features["d_batch_size"])
	cell_output, cell_state = d_cell(d_cell_initial)
	logits = cell_output[0]

	# Visualisations of what attention is doing
	taps = {
		key: cell_output[idx+1] for idx, key in enumerate(d_cell.get_taps().keys())
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
