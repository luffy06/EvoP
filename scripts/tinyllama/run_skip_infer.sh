PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LOG_PATH=$PROJECT_DIR/logs
BLOCKS=(3 5)

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <nsamples> <nclusters> <device> <dataset>"
    NSAMPLES=5
    NCLUSTERS=5 
    DEVICE=0
    DATASET=wikitext2
    echo "Warning: Using default nsamples [$NSAMPLES], nclusters [$NCLUSTERS], device [$DEVICE], dataset [$DATASET]"
else
    NSAMPLES=$1
    NCLUSTERS=$2
    DEVICE=$3
    DATASET=$4
fi

DATA_LOG_PATH=$LOG_PATH/data_sem_cluster_$NSAMPLES\_$NCLUSTERS.json
ALGO_LOG_PATH=$LOG_PATH/algo_gene.json
RESULT_DIR=$PROJECT_DIR/results-$NSAMPLES-$DATASET/tinyllama/skip
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