PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATASET=wikitext2

if [ "$#" -ne 3 ]; then
    echo "Warning: Using default settings search [100%] ratio, 5 samples, device 0"
    RATIO=1
    NSAMPLES=5
    DEVICE=0
else
    RATIO=$1
    NSAMPLES=$2
    DEVICE=$3
fi

DATA_LOG_PATH=$PROJECT_DIR/logs/data_sleb_$NSAMPLES.json
RESULT_DIR=$PROJECT_DIR/motivation-1-$NSAMPLES/tinyllama/ideal
RESULT_DIR=$RESULT_DIR\_$(echo "$RATIO * 100 / 1" | bc | cut -d'.' -f1)

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/skip/skip_ideal.py \
                        --model_name_or_path $MODEL_PATH \
                        --dataset $DATASET \
                        --data_log_path $DATA_LOG_PATH \
                        --batch_size 1 \
                        --ratio $RATIO \
                        --output_dir $RESULT_DIR \
                        > $RESULT_DIR/skip_ideal.log 2>&1