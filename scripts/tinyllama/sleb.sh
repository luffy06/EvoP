PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
NBLOCKS=22
BLOCKS=(0 3 5)

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
RESULT_DIR=$PROJECT_DIR/results-$NSAMPLES-$DATASET/tinyllama/sleb

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

echo "" > $RESULT_DIR/sleb_results.txt
echo "" > $RESULT_DIR/sleb.log
for i in ${BLOCKS[@]};
do
    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/comparisons/sleb/sleb.py \
                            --model_name $MODEL_PATH \
                            --dataset $DATASET \
                            --num_blocks $NBLOCKS \
                            --num_remove_blocks $i \
                            --seed 31 \
                            --data_log_path $LOG_PATH \
                            --result_folder $RESULT_DIR \
                            --eval_ppl \
                            --eval_zeroshot \
                            >> $RESULT_DIR/sleb.log 2>&1
done