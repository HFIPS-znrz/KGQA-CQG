# KGQA-CQG:clarification question generation for knowledge graph question answering
## 环境配置
### 候选答案选择模块
python3.8 <br>
第三方库：./code/EmbedKGQA/requirements.txt
### 澄清问题生成模块
python3.7 <br>
第三方库：./code/generation/requirements.txt

## 候选答案选择
### 1.知识图谱嵌入
安装libkge (https://github.com/uma-pi1/kge) <br>
cd ./code/EmbedKGQA/kge <br>
kge start examples/config_corn.yaml <br>
生成kg_embeddings, 将Checkpoint_best.pt存储到./code/EmbedKGQA/pretrained_models/embeddings/checkpoint_best.pt中 <br>

### 2.关系修剪
cd ./code/EmbedKGQA/KGQA/RoBERTa/, 运行python pruning_main.py 训练关系修剪模型, 生成的文件：./code/EmbedKGQA/KGQA/RoBERTa/checkpoints/pruning

### 3.训练
python main.py --mode train --relation_dim 200 --do_batch_norm 1 --gpu 0 --freeze 1 --batch_size 32 --validate_every 10 --hops webqsp_half --lr 0.00002 --entdrop 0.0 --reldrop 0.0 --scoredrop 0.0 --decay 1.0 --model ComplEx --patience 20 --ls 0.05 --l3_reg 0.001 --nb_epochs 200 --outfile corn_out_complex

### 4.评估
python main.py --mode eval --relation_dim 200 --do_batch_norm 1 --gpu 0 --validate_every 10 --model ComplEx --load_from best_score_model

## 澄清问题生成
### 1.数据预处理
python preprocess.py -train_src ../../../data/single-turn/task2/src-train.txt -train_tgt ../../../data/single-turn/task2/tgt-train.txt -valid_src ../../../data/single-turn/task2/src-test.txt -valid_tgt ../../../data/single-turn/task2/tgt-test.txt -save_data ../../../data/single-turn/task2/demo -dynamic_dict -share_vocab

### 2.模型训练
python train.py -data ../../../data/single-turn/task2/demo -save_model available_models/demo-single-model-transformer13 -gpu_ranks 0 -layers 4 -rnn_size 768 -word_vec_size 768 -transformer_ff 128 -heads 8  -encoder_type transformer -decoder_type transformer -position_encoding -dropout 0.1 -batch_size 16 -accum_count 2 -optim adam -adam_beta2 0.998 -decay_method noam -learning_rate 2 -max_grad_norm 0 -param_init 0  -param_init_glorot -label_smoothing 0.1  -valid_step 10000 -train_steps 40000 -save_checkpoint_steps 40000 -copy_attn

### 3.推理
python translate.py -model available_models/demo-single-model-transformer1_step_40000.pt -src ../../../data/single-turn/task2/src-test.txt -output output.txt -gpu 0 -beam_size 1 <br>
python merge.py

### 4.评估
cd /KGQA-CQG/code/generation/OpenNMT/tools/  <br>
python evaluate.py
