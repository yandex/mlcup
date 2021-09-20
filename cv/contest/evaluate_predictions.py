"""This is a very simple code for evaluation
of zero-shot classification accuracy.
"""

# generic imports
import json
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_file', help='Path to file with ground truth labels')
    parser.add_argument('--predicts_file', help='Path to file with predictions')
    parser.add_argument('--strict', type=int, help='Make sure predictions for all smaples are present')
    parser.add_argument('--average', type=int, help='Output average accuracy across datasets')
    return parser.parse_args()


def main(
    gt_file: str,
    predicts_file: str,
    strict: bool,
    average: bool
):
    gt = json.load(open(gt_file))
    pred = json.load(open(predicts_file))

    assert set(gt.keys()) == set(pred.keys()), \
	    "Some of the dataset keys are missing in the preditions: " + str((gt.keys()) ^ set(pred.keys()))

    results = dict()
    for dataset in gt.keys():
        if strict:
            assert set(gt[dataset].keys()) == set(pred[dataset].keys()), \
			   "Some of the images are missing in the predictions: " + str(set(gt[dataset].keys()) ^ set(pred[dataset].keys()))
        keys = list(set(gt[dataset].keys()) & set(pred[dataset].keys()))
        pred_list = [pred[dataset][x] for x in keys]
        gt_list = [gt[dataset][x] for x in keys]
        accuracy = sum((x == y) for x, y in zip(pred_list, gt_list)) / len(pred_list)
        results[dataset] = accuracy * 100

    if average:
        accuracy = sum(results.values()) / len(results)
        print(accuracy)
    else:
        print(format(results))


if __name__ == "__main__":
    args = parse_arguments()
    main(
        gt_file=args.gt_file,
        predicts_file=args.predicts_file,
        strict=bool(args.strict),
        average=bool(args.average)
    )
