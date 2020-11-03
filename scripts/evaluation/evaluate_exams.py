import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Set

import jsonlines
import numpy as np


class Result:
    def __init__(self):
        self.correct = 0
        self.total = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total

    def toJSON(self) -> Dict[str, Any]:
        return {"accuracy": self.accuracy, "correct": self.correct, "total": self.total}

    def __str__(self):
        return str(self.accuracy)

    def __repr__(self):
        return json.dumps(self.toJSON(), indent=2)


def update_statistics(result: Result, is_correct: bool) -> None:
    """
    Updates the count of total (+1), and correct (+1 if correct) in a given result holder object.
    :param result: A result holder object
    :param is_correct: If the current
    """
    result.total += 1
    result.correct += int(is_correct)


def eval_exams(
    dataset: Dict[str, Dict], predictions: Dict[str, List], granularity: Set[str]
) -> Dict[str, Any]:
    """
    Evaluates the predictions given the provided dataset, and exports the metrics as a dictionary.

    The granularity of the evaluation is controlled with the `granularity` paramer.
    :param dataset: A dict with all the questions, the keys are the ids
    :param predictions: A dict with the predictions, they keys are the ids,
    the values are list of probabilities for each choice
    :param granularity: A set of fine-grained evaluations to include.
    :return: Returns a Dict with the evaluation metrics.
    """
    if "all" in granularity:
        granularity.add("subject")
        granularity.add("language")
        granularity.add("subject_and_language")

    exams_eval = OrderedDict({"overall": Result()})

    fine_eval = {
        "subject": defaultdict(Result),
        "language": defaultdict(Result),
        "subject_and_language": defaultdict(lambda: defaultdict(Result)),
    }

    for id_, prediction in predictions.items():
        info = dataset[id_]["info"]
        answer_key = ord(dataset[id_]["answerKey"]) - ord("A")
        predicted_key = np.argmax(prediction)
        is_correct = answer_key == predicted_key

        subject = info["subject"]
        language = info["language"]

        overall = exams_eval["overall"]
        update_statistics(overall, is_correct)

        if "subject" in granularity:
            update_statistics(fine_eval["subject"][subject], is_correct)

        if "language" in granularity:
            update_statistics(fine_eval["language"][language], is_correct)

        if "subject_and_language" in granularity:
            update_statistics(fine_eval["subject_and_language"][subject][language], is_correct)

    exams_eval["overall"] = exams_eval["overall"].toJSON()

    if granularity:
        exams_eval["fine_grained"] = json.loads(
            json.dumps(
                {k: v for k, v in fine_eval.items() if k in granularity},
                default=lambda x: x.__dict__ if not isinstance(x, Result) else x.toJSON(),
                sort_keys=True,
            )
        )

    return exams_eval


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions_path", type=str, required=True, help="Path to the predictions file.",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the subset that we are evaluating.",
    )

    parser.add_argument(
        "--granularity",
        default="",
        type=str,
        nargs="+",
        choices=["subject", "language", "subject_and_language", "all"],
        help="Granularity options (default: %(default)s)",
    )

    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        required=False,
        help="The output file where the evaluation results will be written. "
        "If no file is provided, then the scripts will use stdout.",
    )

    args = parser.parse_args()
    with open(args.predictions_path, "r") as fp:
        predictions = json.load(fp)

    with jsonlines.open(args.dataset_path) as reader:
        dataset = {q["id"]: q for q in reader}

    exams_eval_json = eval_exams(dataset, predictions, set(args.granularity))

    if args.output_path:
        with open(args.output_path, "w") as fp:
            json.dump(exams_eval_json, fp, indent=4)
    else:
        sys.stdout.write(json.dumps(exams_eval_json, indent=4) + "\n")


if __name__ == "__main__":
    main()
