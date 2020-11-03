import argparse
import pathlib

import jsonlines
from tqdm.auto import tqdm


def read_contexts(contexts_path):
    contexts = {}
    with jsonlines.open(contexts_path) as reader:
        for hit in tqdm(reader, desc="Loading contexts"):
            contexts.update(hit)
    return contexts


def export_with_paras(subset_path, contexts):
    data_with_para = []
    with jsonlines.open(subset_path) as reader:
        for question in tqdm(reader, desc=f"Resolving para for {subset_path}"):
            q_id = question["id"]
            q_hits = contexts[q_id]
            for choice in question["question"]["choices"]:
                choice_label = choice["label"]
                choice_hits = q_hits[choice_label]

                choice["para"] = " ".join([h["hit"]["text"] for h in choice_hits])

            data_with_para.append(question)

    output_file = subset_path.parent / f"{subset_path.stem}_with_para.jsonl"
    print(output_file)
    with jsonlines.open(output_file, "w",) as writer:
        writer.write_all(data_with_para)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--contexts_path", type=pathlib.Path, required=True, help="Path to the resolved contexts.",
    )

    parser.add_argument(
        "--testbed_path",
        type=pathlib.Path,
        required=True,
        help="Path to the testbed, e.g., multilingual, cross-lingual.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        help="The name of the jsonl subset, e.g., train, dev, test.",
    )

    args = parser.parse_args()

    contexts = read_contexts(args.contexts_path)

    subset_path = args.testbed_path / f"{args.subset}.jsonl"
    export_with_paras(subset_path, contexts)


if __name__ == "__main__":
    main()
