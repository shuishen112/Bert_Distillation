# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.
export CUDA_VISIBLE_DEVICES=0

TRANSFER_MODEL=/data/ceph/zhansu/embedding/tanda_bert_base_asnq_torch
GENERAL_TINYBERT_DIR=/data/ceph/zhansu/embedding/General_TinyBERT_4L_312D

TASK_DIR=data/trec
TASK_NAME=mrpc
TMP_TINYBERT_DIR=student_output_first_trec


python3 task_distill.py --teacher_model ${TRANSFER_MODEL} \
                       --student_model ${GENERAL_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TMP_TINYBERT_DIR} \
                       --max_seq_length 256 \
                       --train_batch_size 12 \
                       --num_train_epochs 5 \
                       --do_lower_case



#TINYBERT_DIR=./student_output_second_trec
#python3 task_distill.py --pred_distill  \
#                       --teacher_model ${TRANSFER_TREC} \
#                       --student_model ${TMP_TINYBERT_DIR} \
#                       --data_dir ${TASK_DIR} \
#                       --task_name ${TASK_NAME} \
#                       --output_dir ${TINYBERT_DIR} \
#                       --do_lower_case \
#                       --learning_rate 3e-5  \
#                       --num_train_epochs  5  \
#                       --eval_step 100 \
#                       --max_seq_length 256 \
#                       --train_batch_size 12
