#!/bin/bash

# These scripts all use git commit as the model directory prefix
# this means you can alter your code, commit, then run training
# and TensorBoard will show model performance by git commit.

# This is really helpful as you can always easily reproduce past 
# results, assuming you have the same data.

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train --model-dir output/$COMMIT $@