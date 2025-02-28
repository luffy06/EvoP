PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
BLOCKS=(11)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <nclusters> <device>"
    NCLUSTERS=5
    DEVICE=0
    echo "Warning: Using default nclusters [$NCLUSTERS], default device [$DEVICE]"
else
    NCLUSTERS=$1
    DEVICE=$2
fi

DATA_LOG_PATH=$PROJECT_DIR/logs/data_sem_cluster_ret_$NCLUSTERS.json
ALGO_LOG_PATH=$PROJECT_DIR/logs/algo_gene.json
RESULT_DIR=$PROJECT_DIR/motivation-2-$NCLUSTERS/tinyllama/gene

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

LOG_FILE=$RESULT_DIR/skip.log
echo '' > $LOG_FILE
for i in ${BLOCKS[@]}; 
do
    CUDA_VISIBLE_DEVICES=$DEVICE OPENBLAS_NUM_THREADS=1 python $PROJECT_DIR/src/skip/skip_infer.py \
                            --dataset wikitext2 \
                            --seed 31 \
                            --model_name_or_path $MODEL_PATH \
                            --output_dir $RESULT_DIR \
                            --data_log_path $DATA_LOG_PATH \
                            --algo_log_path $ALGO_LOG_PATH \
                            --nrm_blocks $i \
                            --eval_ppl \
                            >> $LOG_FILE 2>&1
done
