python3 preprocess.py -train_src ../../../data/single-turn/task2/src-train.txt -train_tgt ../../../data/single-turn/task2/tgt-train.txt -valid_src ../../../data/single-turn/task2/src-test.txt -valid_tgt ../../../data/single-turn/task2/tgt-test.txt -save_data ../../../data/single-turn/task2/demo -dynamic_dict -share_vocab

python3 train.py -data ../../../data/single-turn/task2/demo -save_model available_models/demo-single-model-transformer13 -gpu_ranks 0 -layers 4 -rnn_size 768 -word_vec_size 768 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1  -valid_step 10000 -train_steps 40000 -save_checkpoint_steps 40000 -copy_attn

# 注意batch_size默认是1要改
python translate.py -model available_models/demo-single-model-transformer1_step_40000.pt -src ../../../data/single-turn/task2/src-test.txt -output output.txt -replace_unk -verbose -gpu 0 -beam_size 1



CUDA_VISIBLE_DEVICES=0 nohup python3 train.py -data single-turn-data/demo -save_model available_models/demo-single-model-transformer -gpu_ranks 0 -layers 2 -rnn_size 32 -word_vec_size 32 -transformer_ff 32 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1  -valid_step 1000 -copy_attn > log_single_transformer_copy.txt & 6680


CUDA_VISIBLE_DEVICES=4 nohup python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer -gpu_ranks 0 -layers 1 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -copy_attn > log_multi_transformer_copy.txt & 15399



CUDA_VISIBLE_DEVICES=3 nohup python3 train.py -data single-turn-data/demo -save_model available_models/demo-single-model-transformer-without-copy -gpu_ranks 0 -layers 2 -rnn_size 32 -word_vec_size 32 -transformer_ff 32 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1  -valid_step 1000 > log_single_transformer.txt & 11203


CUDA_VISIBLE_DEVICES=2 nohup python3 train.py -data multi-turn-data/demo -save_model available_models/demo-model-transformer-without-copy -gpu_ranks 0 -layers 2 -rnn_size 128 -word_vec_size 128 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -share_vocab > log_multi_transformer.txt & 8599




CUDA_VISIBLE_DEVICES=7 nohup python translate.py -model available_models/demo-single-model-transformer-without-copy_step_10000.pt -src single-turn-data/src-test.txt -output multi-without-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1 > log_test1.txt & 23693
CUDA_VISIBLE_DEVICES=7 nohup python translate.py -model available_models/demo-single-model-transformer_step_5000.pt -src single-turn-data/src-test.txt -output multi-without-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1 > log_test1.txt & 23693
CUDA_VISIBLE_DEVICES=7 nohup python translate.py -model available_models/demo-model-transformer-without-copy_step_10000.pt -src multi-turn-data/src-test.txt -output multi-without-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1 > log_test1.txt & 23693
CUDA_VISIBLE_DEVICES=7 nohup python translate.py -model available_models/demo-model-transformer_step_20000.pt -src multi-turn-data/src-test.txt -output multi-without-copy.txt -replace_unk -verbose -gpu 0 -beam_size 1 > log_test1.txt & 23693

 

perl tools/multi-bleu.perl /data/v-jinxu/code/OpenNMT-py-master_ord/OpenNMT-py-master/single-turn-data/tgt-test1.txt < final_output
perl tools/multi-bleu.perl /data/v-jinxu/code/OpenNMT-py-master_ord/OpenNMT-py-master/multi-turn-data/tgt-test1.txt < final_output
 
python -m debugpy --listen 1234 --wait-for-client train.py -data ../../../data/single-turn/task2/demo -save_model available_models/demo-single-model-transformer9 -gpu_ranks 0 -layers 1 -rnn_size 768 -word_vec_size 768 -transformer_ff 128 -heads 8 -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1 -valid_step 1000 -train_steps 40000 -save_checkpoint_steps 10000 -copy_attn

python -m debugpy --listen 1234 --wait-for-client translate.py -model available_models/demo-single-model-transformer13_step_10000.pt -src ../../../data/single-turn/task2/src-test.txt -output output.txt -replace_unk -verbose -gpu 0 -beam_size 1