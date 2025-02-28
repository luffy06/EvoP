PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATASET=wikitext2
LOG_PATH=$PROJECT_DIR/logs
BLOCKS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <nsamples> <device>"
    NSAMPLES=5
    DEVICE=0
    echo "Warning: Using default settings [5 samples, device 0]"
else
    NSAMPLES=$1
    DEVICE=$2
fi

DATA_LOG_PATH=$LOG_PATH/data_sleb_$NSAMPLES.json
ALGO_LOG_PATH=$LOG_PATH/algo_gene.json
RESULT_DIR=$PROJECT_DIR/motivation-1-$NSAMPLES/tinyllama/gene
if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi
LOG_FILE=$RESULT_DIR/skip.log

echo '' > $LOG_FILE
for i in ${BLOCKS[@]};
do
    LOCAL_RESULT_DIR=$RESULT_DIR/sparsity-$i
    if [ ! -d $LOCAL_RESULT_DIR ]; then
        mkdir -p $LOCAL_RESULT_DIR
    fi
    CUDA_VISIBLE_DEVICES=$DEVICE OPENBLAS_NUM_THREADS=1 python src/skip/skip_infer.py \
                            --dataset $DATASET \
                            --model_name_or_path $MODEL_PATH \
                            --output_dir $LOCAL_RESULT_DIR \
                            --data_log_path $DATA_LOG_PATH \
                            --algo_log_path $ALGO_LOG_PATH \
                            --nrm_blocks $i \
                            --eval_ppl \
                            >> $LOG_FILE 2>&1
done