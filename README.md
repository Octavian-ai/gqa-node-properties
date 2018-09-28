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

The neural network has a series of steps:
- Tokens are embedded into a word-embedding space
- A biLSTM processes the sentence and returns a per-word vector and an overall sentence vector
- A `control cell` performs attention over these word vectors and focusses on one, the `control signal`
- A `read cell` takes this `control signal` and uses it to perform a database lookup. Specifically, this is done by using content-addressed attention on the table of nodes, then performing position-addressed attention on the resulting node data to extract the desired property
- Finally, some dense layers tidy up and transform the output into a word token for output


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

`clevr-graph$ ./generate-station-properties.sh`

You can then compile that into TF records:

`python -m macgraph.input.build --gqa-path gqa-sa-small-100k.yaml --input-dir ./input_data/my_build`

You can download pre-built data here:

### Visualising the predictions

`python -m macgraph.predict --model-dir ./output/my_model`