python3 predict.py \
  --ckpt_path checkpoints/baseline.ckpt \
  --data_directory $1 --predicts_directory $2 \
  ${@:3}