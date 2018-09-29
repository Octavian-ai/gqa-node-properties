# Graph Question Answering: Node properties

This codebase performs a basic Graph-Question-Answer (GQA) task: recalling node properties. 

The dataset is a synthetically generated set of GQA tuples, where each graph is an [imaginary transit network](https://github.com/Octavian-ai/clevr-graph) and each question asks about a property of a particular station in that network. For simplicity, stations have been named with random integers. For example,

> What type of music plays at 3?

Answer:

> Pop

Whilst this sort of property recall is trivial to perform in a database query language, we introduce two challenges:
 - The questions are posed in English, not a query language
 - The recall system is a neural network (i.e. a differentiable function)

## How the system works

The system is a pure (deep) neural network implemented in TensorFlow. It takes a tokenized natural language string as the input, and returns a single word token as output.

See [our medium article](https://medium.com/@DavidMack/graphs-and-neural-networks-reading-node-properties-2c91625980eb) for an in-depth explanation of how this network works.

The system begins by transforming the input question into integer tokens, which are then embedded as vectors.

Next, the control cell³ performs attention over the token vectors. This produces the control signal that is used by the subsequent cells to guide their actions.

Then the read cell uses the control signal to extract a node from the graph node list. It then extracts one property of that node. This cell will be explained in more detail later.

Finally, the output cell transforms the output of the read cell into an answer token (e.g. an integer that maps to an english word in our dictionary)

This code is a snapshot of [MacGraph](https://github.com/Octavian-ai/mac-graph), simplified down to just this task. The network takes inspiration from the [MACnet](https://arxiv.org/abs/1803.03067) architecture.

## Running

First set up the pre-requisites:

```
pipenv install
pipenv shell
```

### Training

`python -m macgraph.train`

### Building the data

You'll need to get a YAML file from [CLEVR-Graph](https://github.com/Octavian-ai/clevr-graph). 

Either [download our pre-build YAML](https://storage.googleapis.com/octavian-static/download/gqa-node-properties/gqa-sp-small-100k.yaml) or create your own:

`clevr-graph$ ./generate-station-properties.sh`

You can then compile that into TF records:

`python -m macgraph.input.build --gqa-path gqa-sa-small-100k.yaml --input-dir ./input_data/my_build`

We provide [pre-compiled TF records](https://storage.googleapis.com/octavian-static/download/gqa-node-properties/tfrecords.zip) and also, the `train.py` script will automatically download and extract this zip file if it doesn't find any training data.

### Visualising the predictions

`./predict.sh` will run the latest trained model in prediction mode. Alternatively you can run the python script yourself on any model directory you wish:

`python -m macgraph.predict --model-dir ./output/my_model`

