PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
DATASET=wikitext2
LOG_PATH=$PROJECT_DIR/logs
METHODS=(dense wanda sleb slicegpt) # skip-ret)
BLOCKS=(3 5)
SPARSITY=(0.1 0.2)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <result_dir> <device>"
    RESULT_DIR=$PROJECT_DIR/results-latency/tinyllama
    DEVICE=0
    echo "Warning: Using default result_dir [$RESULT_DIR], device [$DEVICE]"
else
    RESULT_DIR=$1
    DEVICE=$2
fi

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi
LOG_FILE=$RESULT_DIR/latency.log

echo '' > $LOG_FILE
for i in ${!BLOCKS[@]}; do
    s=${SPARSITY[$i]}
    b=${BLOCKS[$i]}
    for method in ${METHODS[@]}; do
        LOCAL_RESULT_DIR=$RESULT_DIR/sparsity-$b
        MODEL_NAME=`basename $MODEL_PATH`
        SKIP_PATH=$LOCAL_RESULT_DIR/$MODEL_NAME\_$DATASET\_$b.pkl
        CUDA_VISIBLE_DEVICES=$DEVICE OPENBLAS_NUM_THREADS=1 python $PROJECT_DIR/src/utils/latency.py \
                                --model_name $MODEL_PATH \
                                --method $method \
                                --sparsity $s \
                                --generation \
                                --data_log_path logs/data_sleb_5.json \
                                --retriever_dir $LOCAL_RESULT_DIR/all \
                                --skip_tables $SKIP_PATH \
                                >> $LOG_FILE 2>&1
    done
done