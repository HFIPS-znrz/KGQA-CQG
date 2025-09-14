# KGQA-CQG: clarification question generation for knowledge graph question answering

cd code/generation/OpenNMT

## 数据预处理
python preprocess.py -train_src ../../../data/single-turn/task2/src-train.txt -train_tgt ../../../data/single-turn/task2/tgt-train.txt -valid_src ../../../data/single-turn/task2/src-test.txt -valid_tgt ../../../data/single-turn/task2/tgt-test.txt -save_data ../../../data/single-turn/task2/demo -dynamic_dict -share_vocab

## 模型训练
python train.py -data ../../../data/single-turn/task2/demo -save_model available_models/demo-single-model-transformer13 -gpu_ranks 0 -layers 4 -rnn_size 768 -word_vec_size 768 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1  -valid_step 10000 -train_steps 40000 -save_checkpoint_steps 40000 -copy_attn

## 推理
python translate.py -model available_models/demo-single-model-transformer1_step_40000.pt -src ../../../data/single-turn/task2/src-test.txt -output output.txt -gpu 0 -beam_size 1

## 将关键词和框架融合成澄清问句
python merge.py

## 评估
cd /KGQA-CQG/code/generation/OpenNMT/tools/ <br>
python evaluate.py
