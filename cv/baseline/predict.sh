PYTHONPATH=./py_packages:$PYTHONPATH python3 predict.py \
  --ckpt_path checkpoints/baseline_001_inference_fp16.ckpt \
  --data_directory $1 --predicts_file $2
