# CV
This is a baseline solution for CV task in Yandex ML Cup 2021 competition.


# Training
To run training, you have to setup prdinary PyTorch environment with some extra
libraries which can be installed by `pip` or `conda` in a virual environment.
For full list of required libraries see `baseline/requirements.txt`.

To train the baseline model, you need to have at least single GPU.

Baseline training can be run by set of commands similar to
```bash
cd baseline
python3 -m venv venv
source venv/bin/activate
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install -r requirements.txt
CUDA_VISIBLE_DEVICES=1,2 python train.py +name=baseline train.trainer_params.gpus=2 _data.paths.images_directory='/home/${oc.env:USER}/path/to/images/dir/' _data.paths.metadata_file='/home/${oc.env:USER}/path/to/metadata.json'
```

# Pretrained checkpoint
For convienience, pretrained model checkpoint can be downloaded by running
`bash download_checkpoints.sh`


# Making predictions on sample data
A tiny subset of public data is avaialble at `contest/data` directory.
`contest/data/public_subset/` directory contains two datasets, with classes labels 
and source images in each of them.

To run zero-shot classes prediction on these datasets, run
```bash
cd baseline
python predict.py --ckpt_path /path/to/checkpoint --data_directory ../contest/data/public_subset/ --predicts_file ../contest/predictions.json
```

You may add `--device cuda` argument to speed up prediction locally.
To approximate evaluation speed on the Yandex Contest server, add `--device cpu --num_threads 1`
arguments to imitate single available CPU.


# Evaluating predictions on sample data
File `contest/data/public_subset_gt.json` contains GT labels. You can use them to calculate accuracy by running
```bash
python ../contest/evaluate_predictions.py --gt_file ../contest/data/public_subset_gt.json --predicts_file ../contest/predictions.json --average 1 --strict 1
```
It will print an accuracy like `69.2057142857143`


# Prepare checkpoint for inference
In order to send the model for online evaluation in Yandex.Contest system,
you need to make it compact and runnable without network connection.
In order to do this you may use script `convert_checkpoint_for_inference.py`
which disables loading pretrained embeddings and weights from network and optionally converts checkpoint
into FP16 mode (which roughly halves the checkpoint size), while also deleting
optimizer states from it.


# Making a submission to Yandex Contest
`setup.sh` and `predict.sh` are the two files used by Y.Contest to run your submission.
`setup.sh` should install come extra required libraries missing in Y.Contest environment and `predict.sh` is run with two arguments: first is path to source data (like `contest/data/eval/public_subset`) and the second is the .json file where prediction should be put.

To make a submission, simply archive whole `baseline` directory and send it to Y.Contest system for online evaulation:
```bash
cd baseline
zip -r ../submission.zip ./*
```
