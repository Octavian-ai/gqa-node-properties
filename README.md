# Graph Question Answering: Node properties


## Running

First set up the pre-requisites:

```
pipenv install
pipenv shell
```

### Training

`python -m macgraph.train`

### Building the data

You'll need to get a YAML file from [CLEVR-Graph](). You can then build it into TF records:

`python -m macgraph.input.build --gqa-path gqa.yaml --input-dir ./input_data/my_build`

### Visualising the predictions

`python -m macgraph.predict --model-dir ./output/my_model`