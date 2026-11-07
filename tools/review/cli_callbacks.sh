#!/bin/bash
# Parse lite_ladder.yaml + the train_ladder.sh command line with LightningCLI (--print_config, no instantiation) and
# show the resulting trainer.callbacks, LR scheduler and denoiser dims.
cd "$(dirname "$0")/../../baselines/PixelGen"
LAST=16125; MS=129000
CB='[{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"resume-{step:06d}","monitor":null,"save_last":true,"save_top_k":0,"every_n_train_steps":'$LAST'}},{"class_path":"src.callbacks.model_checkpoint.CheckpointHook","init_args":{"filename":"ms-{step:06d}","monitor":null,"save_last":false,"save_top_k":-1,"every_n_train_steps":'$MS'}}]'
python main.py fit --config configs_h100/lite_ladder.yaml --trainer.max_steps 516000 --trainer.callbacks "$CB" \
  --model.denoiser.init_args.hidden_size 640 --model.denoiser.init_args.num_groups 10 \
  --model.denoiser.init_args.num_blocks 9 --model.denoiser.init_args.num_text_blocks 2 \
  --tags.exp ladder_d640 --trainer.logger.init_args.name ladder_d640 --trainer.logger.init_args.id ladder-d640 \
  --trainer.logger.init_args.id ladder-smoke-x --print_config 2>/dev/null | sed -n "/^# lightning\|^seed_everything/,\$p" | python -c '
import sys, yaml
c = yaml.safe_load(sys.stdin)
for cb in c["trainer"]["callbacks"]: print("callback", cb["class_path"].rsplit(".",1)[-1], {k: cb["init_args"].get(k) for k in ("filename","save_last","save_top_k","every_n_train_steps","save_on_train_epoch_end")})
print("logger id", c["trainer"]["logger"]["init_args"]["id"])
print("lr", c["model"]["lr_scheduler"]["init_args"])
d = c["model"]["denoiser"]["init_args"]; print("denoiser", {k: d[k] for k in ("hidden_size","num_groups","num_blocks","num_text_blocks","txt_max_length")})
print("max_steps", c["trainer"]["max_steps"], "check_val_every_n_epoch", c["trainer"]["check_val_every_n_epoch"])
'
