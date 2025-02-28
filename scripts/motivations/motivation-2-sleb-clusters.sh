PROJECT_DIR=$(git rev-parse --show-toplevel)
MODEL_PATH=TinyLlama/TinyLlama-1.1B-Chat-v1.0
NBLOCKS=22
BLOCKS=(11)

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <nclusters> <device>"
    NCLUSTERS=5
    DEVICE=0
else
    NCLUSTERS=$1
    DEVICE=$2
fi

LOG_PATH=$PROJECT_DIR/logs/data_sem_cluster_ret_$NCLUSTERS.json
RESULT_DIR=$PROJECT_DIR/motivation-2-$NCLUSTERS/tinyllama/sleb

if [ ! -d $RESULT_DIR ]; then
    mkdir -p $RESULT_DIR
fi

echo "" > $RESULT_DIR/sleb.log
rm -rf $RESULT_DIR/cluster-*
for i in ${BLOCKS[@]};
do
    CUDA_VISIBLE_DEVICES=$DEVICE python $PROJECT_DIR/src/comparisons/sleb/sleb_cluster.py \
                            --model_name $MODEL_PATH \
                            --num_blocks $NBLOCKS \
                            --num_remove_blocks $i \
                            --seed 31 \
                            --data_log_path $LOG_PATH \
                            --result_folder $RESULT_DIR \
                            --result_file sleb_results.txt \
                            --eval_ppl \
                            >> $RESULT_DIR/sleb.log 2>&1
done