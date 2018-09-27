#!/bin/bash

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)
LATEST=$(ls -td -- output/* | head -n 1)

python -m macgraph.predict --model-dir $LATEST $@