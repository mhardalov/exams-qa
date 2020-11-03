import argparse
import os
from functools import partial
from multiprocessing.pool import Pool

from elasticsearch import Elasticsearch
from jsonlines import jsonlines
from tqdm.auto import tqdm

from examsqa.mappings import index_mapping


def query_es_bulk(
    question,
    host,
    port,
    num_hits=25,
    query_field=("text",),
    highligh_size=400,
    num_highlights=3,
    explain=False,
):
    elastic = Elasticsearch(host, port=port)

    def map_response(response):
        return [
            {
                "score": x["_score"],
                "hit": {
                    "title": x["_source"]["title"],
                    "text": x["_source"]["text"],
                    "url": x["_source"]["isbn"],
                },
            }
            for x in response["hits"]["hits"]
        ]

    question_data = question["question"]

    if not question_data["choices"]:
        return None
    index_name = index_mapping[question["info"]["language"].lower()]

    body = []
    for choice in question_data["choices"]:
        body.append({"index": index_name})
        query = " ".join((question_data["stem"], choice["text"]))
        body.append(
            {
                "explain": explain,
                "query": {"multi_match": {"query": query, "fields": query_field}},
                "highlight": {
                    "fragment_size": highligh_size,
                    "type": "plain",
                    "number_of_fragments": num_highlights,
                    "fields": {"passage": {}},
                },
                "from": 0,
                "size": num_hits,
            }
        )

    responses = elastic.msearch(index=index_name, body=body, request_timeout=60)
    return [map_response(res) for res in responses["responses"]]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("host", help="Elasticsearch host.")
    parser.add_argument("-p", "--port", default=9200, help="port, default is 9200", type=int)

    parser.add_argument(
        "--question_paths", nargs="+", type=str, required=True, help="Path to the questions."
    )

    args = parser.parse_args()
    host = args.host
    port = args.port

    for path in args.question_paths:
        updated_questions = list()
        resolved_contexts = []
        raw_contexts = []
        f_context = partial(query_es_bulk, host=host, port=port)

        with jsonlines.open(path) as reader:
            questions = list(reader)

            for result in tqdm(Pool().imap(f_context, questions), total=len(questions)):
                resolved_contexts.append(result)

            for question, per_choice_contexts in zip(questions, resolved_contexts):
                raw_context = {}
                q_updated = question.copy()

                for choice, contexts in zip(question["question"]["choices"], per_choice_contexts):
                    choice["para"] = " ".join([c["hit"]["text"] for c in contexts])
                    raw_context[choice["label"]] = contexts

                updated_questions.append(q_updated)
                raw_contexts.append({question["id"]: raw_context})

        base_dir = os.path.dirname(path)
        base = os.path.basename(path)
        name = os.path.splitext(base)[0]

        with jsonlines.open(os.path.join(base_dir, name + "_with_hits.jsonl"), "w") as writer:
            writer.write_all(raw_contexts)

        with jsonlines.open(os.path.join(base_dir, name + "_with_para.jsonl"), "w") as writer:
            writer.write_all(updated_questions)


if __name__ == "__main__":
    main()
