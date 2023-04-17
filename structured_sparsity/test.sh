#!/bin/bash

set +x 
set +e

echo '--------------------------------------------------------------------------------'
echo TensorFlow Container $NVIDIA_TENSORFLOW_VERSION
echo Container Build ID $NVIDIA_BUILD_ID
echo Uptime: $(uptime)
echo '--------------------------------------------------------------------------------'

# Add a script dir here for the resnet50 that supports sparsity.
SCRIPT_DIR="/home/workspace/image_zoo/resnet50_sparsity/nvidia-examples/cnn"
echo Testing the resnet50 scripts from $SCRIPT_DIR

GPUS=$(nvidia-smi -L 2>/dev/null | grep "^GPU" | wc -l || echo 1)
BATCH_SIZE=128
PRECISION="fp16"
DATA="--data_dir=/data/imagenet/train-val-tfrecord-480"

CUDACC=$(/usr/local/bin/deviceQuery | grep -i 'cuda capability' | awk '{print $6}' | head -1)
MAJORCC="${CUDACC%.*}"
if [[ $MAJORCC -lt 7 ]]; then
    INPUT_FORMAT="--image_format=channels_first"
fi

launcher() {
    local TMP_MOD_DIR="$1"
    SCRIPT="$2"
    EXTRA_OPTS=""
    if [[ "$#" -eq  "3" && "$3" == "sparsity" ]]; then
        EXTRA_OPTS="--enable_sparsity"
    fi

    if [[ "$EXTRA_OPTS" == "--enable_sparsity" ]]; then
      echo Testing the saved model in $SCRIPT with sparsity
    else
      echo Testing the saved model in $SCRIPT 
    fi
    mpiexec --allow-run-as-root -np $GPUS python -u \
        $SCRIPT_DIR/$SCRIPT \
        $DATA $INPUT_FORMAT $EXTRA_OPTS \
        --num_iter=101 \
        --iter_unit=batch \
        --display_every=50 \
        --export_dir=$TMP_MOD_DIR \
        --batch_size=$BATCH_SIZE \
        --precision=$PRECISION &> log.tmp
    RET=$?

    if [[ $RET -ne 0 ]]; then
        cat log.tmp
        echo SAVED MODEL EXPORT SCRIPT FAILED FOR $SCRIPT
        exit 1
    fi

    python -u \
        $SCRIPT_DIR/$SCRIPT \
        --export_dir=$TMP_MOD_DIR \
        --batch_size=$BATCH_SIZE \
        --predict $EXTRA_OPTS &> log.tmp
    RET=$?

    if [[ $RET -ne 0 ]]; then
        cat log.tmp
        echo SAVED MODEL PREDICTION SCRIPT FAILED FOR $SCRIPT
        exit 1
    fi

    if [[ "$EXTRA_OPTS" == "--enable_sparsity" ]]; then
      PRUNE_RESULTS=$(grep -i '\[TF-ASP\]' log.tmp | tail -1 | awk '{print $2}')
      if [[ "$PRUNE_RESULTS" != "53/54" ]]; then
        echo "SAVED MODEL PREDICTION FAILED FOR SPARSITY: expected '53/54' " \
             "pruned layers, but got '$PRUNE_RESULTS'."
      fi
    fi

    rm log.tmp
}


TMP_MOD_DIR="$(mktemp -d tmp.XXXXXX)"
launcher $TMP_MOD_DIR "resnet_ctl.py" 
launcher $TMP_MOD_DIR "resnet_ctl.py" "sparsity"
rm -rf "$TMP_MOD_DIR"

TMP_MOD_DIR="$(mktemp -d tmp.XXXXXX)"
launcher $TMP_MOD_DIR "resnet.py" 
launcher $TMP_MOD_DIR "resnet.py" "sparsity"
rm -rf "$TMP_MOD_DIR"

echo All tests pass.
exit 0

