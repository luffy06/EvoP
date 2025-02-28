PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
SPARSITY=(0.10 0.20)

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

RESULT_DIR=$PROJECT_DIR/results-$NSAMPLES-$DATASET/tinyllama/slicegpt

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

echo "" > $RESULT_DIR/slicegpt.log
for i in ${SPARSITY[@]}; do
    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/comparisons/slicegpt/run_slicegpt.py \
                            --model $MODEL_PATH \
                            --cal-dataset $DATASET \
                            --seed 31 \
                            --cal-nsamples $NSAMPLES \
                            --cal-max-seqlen 2048 \
                            --sparsity $i \
                            --device cuda \
                            --eval_ppl \
                            --eval_zeroshot \
                            >> $RESULT_DIR/slicegpt.log 2>&1
done
