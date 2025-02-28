PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
BLOCKS=(3 5)

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <result_dir> <device> <dataset>"
    RESULT_DIR=$PROJECT_DIR/results-5-wikitext2/tinyllama/skip
    DEVICE=0
    DATASET=wikitext2
    echo "Error: Must provide result_dir and device"
    exit
else
    RESULT_DIR=$1
    DEVICE=$2
    DATASET=$3
fi

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi
LOG_FILE=$RESULT_DIR/skip_eval.log

echo '' > $LOG_FILE
for i in ${BLOCKS[@]};
do
    MODEL_NAME=`basename $MODEL_PATH`
    SKIP_PATH=$RESULT_DIR/sparsity-$i/$MODEL_NAME\_$DATASET\_$i.pkl
    CUDA_VISIBLE_DEVICES=$DEVICE OPENBLAS_NUM_THREADS=1 python src/skip/skip_eval.py \
                            --model_name_or_path $MODEL_PATH \
                            --seed 31 \
                            --skip_layers $SKIP_PATH \
                            --eval_ppl \
                            --eval_zeroshot \
                            >> $LOG_FILE 2>&1
done