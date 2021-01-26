
transfer_teacher=/root/program/transformers/transfer_trec_20200706_02
TASK_NAME=mrpc
OUTPUT_DIR=./final_output
TASK_DIR=./data/trec
TEACHERBERT_DIR=./data/bert_wiki_ft
python3 task_distill.py --do_eval \
                       --student_model ${transfer_teacher} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${OUTPUT_DIR} \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128
