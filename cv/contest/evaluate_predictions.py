"""This is a very simple code for evaluation
of zero-shot classification accuracy.

Takes predictions from json files in `predicts_directory` and
GT labels from json files in `gt_directory`, and calculates
accuracy for each dataset by comparing labels.
"""

# generic imports
import click
import json
import os
import numpy as np


@click.command()
@click.option('--gt_directory', help='Path to directory with ground truth labels')
@click.option('--predicts_directory', help='Path to directory with predictions')
@click.option('--strict', is_flag=True, help='Make sure predictions for all smaples are present')
@click.option('--average', is_flag=True, help='Output average accuracy across datasets')
def main(
    gt_directory: str,
    predicts_directory: str,
    strict: bool,
    average: bool
):
    datasets = os.listdir(gt_directory)
    results = dict()
    for dataset in datasets:
        with open(f"{predicts_directory}/{dataset}") as f:
            predicts = json.load(f)
        with open(f"{gt_directory}/{dataset}") as f:
            gt = json.load(f)
        gt = {k + '.jpg': v['class'] for k, v in gt.items()}

        if strict:
            assert set(predicts.keys()) == set(gt.keys())
        keys = list(set(predicts.keys()) & set(gt.keys()))
        np.random.shuffle(keys)

        predicts_list = [predicts[x] for x in keys]
        gt_list = [gt[x] for x in keys]
        accuracy = (np.array(predicts_list) == np.array(gt_list)).mean()
        results[dataset] = accuracy * 100

    if average:
        accuracy = np.mean(list(results.values()))
        print(f"Accuracy: {accuracy}")
    else:
        print(f"Accuracies: {results}")


if __name__ == "__main__":
    main()
