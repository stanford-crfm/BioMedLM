import json
import os
import subprocess
import sys

env_setup_cmd = "task=medqa_usmle_hf ; datadir=data/$task ; export WANDB_PROJECT='biomedical-nlp-eval'"

experiments = [json.loads(line) for line in open(sys.argv[1]).read().split("\n") if line]

for experiment in experiments:
    checkpoint = experiment["checkpoint"]
    lr = experiment["lr"]
    epochs = experiment["epochs"]
    grad_accum = experiment["grad_accum"]
    train_per_device_batch_size = experiment["train_per_device_batch_size"]
    num_devices = experiment["num_devices"] if "num_devices" in experiment else 8
    batch_size = int(num_devices) * int(grad_accum) * int(train_per_device_batch_size)
    tokenizer = experiment["tokenizer"]
    numerical_format = experiment["numerical"] if "numerical" in experiment else "bf16"
    seed = experiment["seed"]
    use_flash = experiment["use_flash"]
    run_name = f"{os.path.basename(checkpoint)}-lr={lr}-batch_size={batch_size}-epochs={epochs}-seed={seed}-task=medqa"
    exp_cmd = (
        f"python -m torch.distributed.launch --nproc_per_node={num_devices} --nnodes=1 --node_rank=0"
        f" run_multiple_choice.py --use_flash {use_flash} --tokenizer_name {tokenizer} --model_name_or_path"
        f" {checkpoint} --train_file data/medqa_usmle_hf/train.json --validation_file data/medqa_usmle_hf/dev.json"
        " --test_file data/medqa_usmle_hf/test.json --do_train --do_eval --do_predict --per_device_train_batch_size"
        f" {train_per_device_batch_size} --per_device_eval_batch_size 1 --gradient_accumulation_steps {grad_accum}"
        f" --learning_rate {lr} --warmup_ratio 0.5 --num_train_epochs {epochs} --max_seq_length 512"
        f" --{numerical_format} --seed {seed} --data_seed {seed} --logging_first_step --logging_steps 20"
        f" --save_strategy no --evaluation_strategy steps --eval_steps 500 --run_name {run_name} "
        " --output_dir trash/"
        " --overwrite_output_dir"
    )
    if "sharded_ddp" in experiment and experiment["sharded_ddp"].lower() == "true":
        exp_cmd += " --sharded_ddp zero_dp_2 "
    print("---")
    print(exp_cmd)
    subprocess.call(f"{env_setup_cmd} ; {exp_cmd}", shell=True)
