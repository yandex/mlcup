"""This is a very simple code for evaluation
of zero-shot classification accuracy.
"""

# generic imports
import click
import json
import numpy as np


@click.command()
@click.option('--gt_file', help='Path to file with ground truth labels')
@click.option('--predicts_file', help='Path to file with predictions')
@click.option('--strict', is_flag=True, help='Make sure predictions for all smaples are present')
@click.option('--average', is_flag=True, help='Output average accuracy across datasets')
def main(
    gt_file: str,
    predicts_file: str,
    strict: bool,
    average: bool
):
    gt = json.load(open(gt_file))
    pred = json.load(open(predicts_file))

    results = dict()
    for dataset in gt.keys():
        if strict:
            assert set(gt[dataset].keys()) == set(pred[dataset].keys()), \
                set(gt[dataset].keys()) ^ set(pred[dataset].keys())
        keys = list(set(gt[dataset].keys()) & set(pred[dataset].keys()))
        pred_list = [pred[dataset][x] for x in keys]
        gt_list = [gt[dataset][x] for x in keys]
        accuracy = (np.array(pred_list) == np.array(gt_list)).mean()
        results[dataset] = accuracy * 100

    if average:
        accuracy = np.mean(list(results.values()))
        print(f"Accuracy: {accuracy}")
    else:
        print(f"Accuracies: {results}")


if __name__ == "__main__":
    main()
