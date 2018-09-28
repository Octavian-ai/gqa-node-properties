
import tensorflow as tf
import math

def assert_shape(tensor, shape, batchless=False):

	read_from = 0 if batchless else 1

	lhs = tf.TensorShape(tensor.shape[read_from:])
	rhs = tf.TensorShape(shape)

	lhs.assert_is_compatible_with(rhs)
	
	# assert lhs == shape, f"{tensor.name} is wrong shape, expected {shape} found {lhs}"

def assert_rank(tensor, rank):
	assert len(tensor.shape) == rank, f"{tensor.name} is wrong rank, expected {rank} got {len(tensor.shape)}"


def dynamic_assert_shape(tensor, shape, name=None):
	"""
	Check that a tensor has a shape given by a list of constants and tensor values.

	This function will place an operation into your graph that gets executed at runtime.
	This is helpful because often tensors have many dynamic sized dimensions that
	you cannot otherwise compare / assert are as you expect.

	For example, measure a dimension at run time:
	`batch_size = tf.shape(my_tensor)[0]`
	
	then assert another tensor does indeed have the right shape:  
	`other_tensor = dynamic_assert_shape(other_tensor, [batch_size, 16])`

	You should use this as an inline identity function so that the operation it generates
	gets added and executed in the graph

	Returns: the argument `tensor` unchanged
	"""

	tensor_shape = tf.shape(tensor)
	tensor_shape = tf.cast(tensor_shape, tf.int64)
	
	expected_shape = tf.convert_to_tensor(shape)
	expected_shape = tf.cast(expected_shape, tf.int64)
	
	t_name = "tensor" if tf.executing_eagerly() else tensor.name

	assert_op = tf.assert_equal(tensor_shape, expected_shape, message=f"Asserting shape of {t_name}", summarize=10, name=name)

	with tf.control_dependencies([assert_op]):
		return tf.identity(tensor, name="dynamic_assert_shape")



def minimize_clipped(optimizer, value, max_gradient_norm, var=None):
	global_step = tf.train.get_global_step()

	if var is None:
		var = tf.trainable_variables()
	
	gradients = tf.gradients(value, var)
	clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
	grad_dict = dict(zip(var, clipped_gradients))
	op = optimizer.apply_gradients(zip(clipped_gradients, var), global_step=global_step)
	return op, grad_dict


def deeep(tensor, width, depth=2, residual_depth=3, activation=tf.nn.tanh):
	"""
	Quick 'n' dirty "let's slap on some layers" function. 

	Implements residual connections and applys them when it can. Uses this schematic:
	https://blog.waya.ai/deep-residual-learning-9610bb62c355
	"""
	with tf.name_scope("deeep"):

		if residual_depth is not None:
			for i in range(math.floor(depth/residual_depth)):
				tensor_in = tensor

				for j in range(residual_depth-1):
					tensor = tf.layers.dense(tensor, width, activation=activation)

				tensor = tf.layers.dense(tensor, width)
			
				if tensor_in.shape[-1] == width:
					tensor += tensor_in
			
				tensor = activation(tensor)

			remaining = depth % residual_depth
		else:
			remaining = depth

		for i in range(remaining):
			tensor = tf.layers.dense(tensor, width, activation=activation)

		return tensor




def download_data(args):
	if not tf.gfile.Exists(args["train_input_path"]):
		zip_path = "./tfrecords.zip"
		print("Downloading training data (61mb)")
		urllib.request.urlretrieve ("https://storage.googleapis.com/octavian-static/download/gqa-node-properties/tfrecords.zip", zip_path)

		print("Unzipping...")
		pathlib.Path(args["input_dir"]).mkdir(parents=True, exist_ok=True)
		with zipfile.ZipFile(zip_path,"r") as zip_ref:
			zip_ref.extractall(args["input_dir"])

