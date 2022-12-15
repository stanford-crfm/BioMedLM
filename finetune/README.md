# Biomedical downstream evaluation

## NLU
### Dependencies
```bash
conda create -n pubmedgpt python=3.8.12 pytorch=1.12.1 torchdata cudatoolkit=11.3 -c pytorch
conda activate pubmedgpt
pip install -r setup/requirements.txt
```

### Usage
For PubMedQA, go to `seqcls/` and run the following command:
```bash
task=pubmedqa_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 run_seqcls_gpt.py \
 --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer --model_name_or_path {checkpoint} --train_file \
 $datadir/train.json --validation_file $datadir/dev.json --test_file $datadir/test.json --do_train \
 --do_eval --do_predict --per_device_train_batch_size 1 --gradient_accumulation_steps \
 {grad_accum} --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {num_epochs}  --max_seq_length \
 {seq_len}  --logging_steps 100 --save_strategy no --evaluation_strategy no --output_dir \
 {run_dir} --overwrite_output_dir --bf16
 --seed {seed} --run_name {name}
```

```

For MedQA-USMLE, go to `mc/` and run the following command:
```bash
task=medqa_usmle_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python -m torch.distributed.launch --nproc_per_node={num_devices} --nnodes=1 --node_rank=0 \
 run_multiple_choice.py --use_flash {use_flash} --tokenizer_name {tokenizer} --model_name_or_path \
 {checkpoint} --train_file data/medqa_usmle_hf/train.json --validation_file data/medqa_usmle_hf/dev.json \
 --test_file data/medqa_usmle_hf/test.json --do_train --do_eval --do_predict --per_device_train_batch_size \
 {train_per_device_batch_size} --per_device_eval_batch_size 1 --gradient_accumulation_steps {grad_accum} \
 --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {epochs} --max_seq_length 512 \
 --{numerical_format} --seed {seed} --data_seed {seed} --logging_first_step --logging_steps 20 \
 --save_strategy no --evaluation_strategy steps --eval_steps 500 --run_name {run_name} \
 --output_dir trash/ \
 --overwrite_output_dir 
```
