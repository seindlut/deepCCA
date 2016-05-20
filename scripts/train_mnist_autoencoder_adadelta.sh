
#!/usr/bin/env bash
# This script train a siamese network on the holidays dataset
cd ../

TOOLS=~/opt/caffe/build/tools
ARCHI=$1

$TOOLS/caffe train --solver=archi/$ARCHI/solver.prototxt 2>&1 | tee logs/$ARCHI.log
