#!/bin/bash

COMMIT=$(git --no-pager log --pretty=format:'%h' -n 1)

python -m macgraph.train --model-dir output/$COMMIT