PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
NBLOCKS=22
BLOCKS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <nsamples> <device>"
    NSAMPLES=5
    DEVICE=0
else
    NSAMPLES=$1
    DEVICE=$2
fi

LOG_PATH=$PROJECT_DIR/logs/data_sleb_$NSAMPLES.json
RESULT_DIR=$PROJECT_DIR/motivation-1-$NSAMPLES/tinyllama/sleb

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

echo "" > $RESULT_DIR/sleb_results.txt
echo "" > $RESULT_DIR/sleb.log
for i in ${BLOCKS[@]};
do
    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/comparisons/sleb/sleb.py \
                            --model_name $MODEL_PATH \
                            --num_blocks $NBLOCKS \
                            --num_remove_blocks $i \
                            --seed 31 \
                            --data_log_path $LOG_PATH \
                            --result_folder $RESULT_DIR \
                            --eval_ppl \
                            >> $RESULT_DIR/sleb.log 2>&1
done