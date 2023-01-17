# Biomedical downstream evaluation

## NLU
### Dependencies
```bash
conda create -n pubmedgpt python=3.8.12 pytorch=1.12.1 torchdata cudatoolkit=11.3 -c pytorch
conda activate pubmedgpt
pip install -r setup/requirements.txt
```

### Usage

Note we are not providing the data. Demo versions of the `.jsonl` files are presented to show expected format.
There should be one json per line for each example in the respective data sets for these tasks.

For PubMedQA and BioASQ, go to `seqcls/` and run the following command (change paths appropriately for task):
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


For MedQA-USMLE, go to `mc/` and run the following command:
```bash
task=medqa_usmle_hf
datadir=data/$task
outdir=runs/$task/GPT2
mkdir -p $outdir
python -m torch.distributed.launch --nproc_per_node={num_devices} --nnodes=1 --node_rank=0 \
  run_multiple_choice.py --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer --model_name_or_path \
  {checkpoint} --train_file data/medqa_usmle_hf/train.json --validation_file data/medqa_usmle_hf/dev.json \
  --test_file data/medqa_usmle_hf/test.json --do_train --do_eval --do_predict --per_device_train_batch_size \
  {train_per_device_batch_size} --per_device_eval_batch_size 1 --gradient_accumulation_steps {grad_accum} \
  --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {epochs} --max_seq_length 512 \
  --{numerical_format} --seed {seed} --data_seed {seed} --logging_first_step --logging_steps 20 \
  --save_strategy no --evaluation_strategy steps --eval_steps 500 --run_name {run_name} \
  --output_dir trash/ \
  --overwrite_output_dir 
```

## NLG
Go to `./textgen`.

### Usage (seq2seq tasks)
Make sure the task dataset is in `./textgen/data`. See `meqsum` (a medical text simplification task) as an example. The dataset folder should have `<split>.source` and `<split>.target` files.

Go to `./textgen/gpt2`.
To finetune, run:
```
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  finetune_for_summarization.py --output_dir {run_dir} --model_name_or_path {checkpoint}
  --tokenizer_name stanford-crfm/pubmed_gpt_tokenizer --per_device_train_batch_size 1 
  --per_device_eval_batch_size 1 --save_strategy no --do_eval --train_data_file 
  data/meqsum/train.source --eval_data_file data/meqsum/val.source --save_total_limit 2 
  --overwrite_output_dir --gradient_accumulation_steps {grad_accum} --learning_rate {lr} 
  --warmup_ratio 0.5 --weight_decay 0.0 --seed 7 --evaluation_strategy steps --eval_steps 200 
  --bf16 --num_train_epochs {num_epochs} --logging_steps 100 --logging_first_step 
```

After finetuning, run generation on the test set by:

```
CUDA_VISIBLE_DEVICES=0 python -u run_generation_batch.py --fp16 --max_source_length -1 --length 400 --model_name_or_path={finetune_checkpoint} --num_return_sequences 5 --stop_token [SEP] --tokenizer_name={finetune_checkpoint} --task_mode=meqsum --control_mode=no --tuning_mode finetune --gen_dir gen_results__tgtlen400__no_repeat_ngram_size6 --batch_size 9 --temperature 1.0 --no_repeat_ngram_size 6 --length_penalty -0.5 --wandb_entity=None --wandb_project=None --wandb_run_name=None
```


### Acknowledgement
The NLG part of the code was built on https://github.com/XiangLi1999/PrefixTuning
