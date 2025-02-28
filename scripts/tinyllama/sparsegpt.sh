PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <nsamples> <device> <dataset>"
    NSAMPLES=5
    DEVICE=0
    DATASET=wikitext2
    echo "Warning: Using default nsamples [$NSAMPLES], device [$DEVICE], dataset [$DATASET]"
else
    NSAMPLES=$1
    DEVICE=$2
    DATASET=$3
fi

LOG_PATH=$PROJECT_DIR/logs/data_sleb_$NSAMPLES.json
RESULT_DIR=$PROJECT_DIR/results-$NSAMPLES-$DATASET/tinyllama/sparsegpt

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

echo "" > $RESULT_DIR/sparsegpt.log
CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/comparisons/wanda/wanda.py \
                        --model $MODEL_PATH \
                        --dataset $DATASET \
                        --seed 31 \
                        --data_log_path $LOG_PATH \
                        --sparsity_ratio 0.5 \
                        --sparsity_type '2:4' \
                        --prune_method 'sparsegpt' \
                        --eval_ppl \
                        --eval_zeroshot \
                        >> $RESULT_DIR/sparsegpt.log 2>&1