import argparse
import json
from datetime import datetime
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import regex as re
from elasticsearch import Elasticsearch, helpers
from nltk import tokenize
from tqdm import tqdm


def create_doc(title, url, text, page):
    doc = {
        "title": title,
        "url": url,
        "text": text,
        "page": page,
        "timestamp": datetime.now(),
    }

    return doc


def create_index(es_client, index_name, schema):
    es_client.indices.delete(index=index_name, ignore=[400, 404])
    es_client.indices.create(index=index_name, ignore=400, body=schema)


def push_actions(actions, host, port):
    es_client = Elasticsearch(host, port=port)
    helpers.bulk(es_client, actions, request_timeout=60)


def split_text_paragraph(article):
    paragraphs = [x.strip() for x in re.split(r"\n+", article["text"])]
    return paragraphs


def split_text_sliding_window(text, window, stride):
    paragraphs = [text[i : (i + window)] for i in range(0, len(text), stride)]
    return paragraphs


def file_to_actions(file_path, index_name, strategy, stride, window):
    actions = []
    with open(file_path, "r") as lines:
        for line in lines:
            article = json.loads(line)
            text = article["text"]
            if strategy == "window":
                assert window > 0
                paragraphs = split_text_sliding_window(text, window, stride)
            elif strategy == "paragraph":
                paragraphs = split_text_paragraph(article)
            else:
                paragraphs = tokenize.sent_tokenize(article["text"])

            for text in paragraphs:
                doc = create_doc(article["title"], article["url"], text, int(article["id"]))

                action = {
                    "_index": index_name,
                    "_type": strategy,
                    "_source": doc,
                }
                actions.append(action)

    return actions


def fill_index(wiki_path, host, port, index_name, strategy, window, stride, bulk_size):
    wiki_files = [p for p in Path(wiki_path).rglob("*") if p.is_file()]

    f_actions = partial(
        file_to_actions, index_name=index_name, strategy=strategy, window=window, stride=stride
    )
    f_push_actions = partial(push_actions, host=host, port=port)
    all_actions = []
    for actions in tqdm(Pool().imap(f_actions, wiki_files), total=len(wiki_files)):
        all_actions += actions

    action_splits = np.array_split(all_actions, len(all_actions) // bulk_size + 1)
    for _ in tqdm(map(f_push_actions, action_splits), total=len(action_splits)):
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("index", help="Name of index to create")
    parser.add_argument("host", help="Elasticsearch host.")
    parser.add_argument("-p", "--port", default=9200, help="port, default is 9200", type=int)

    parser.add_argument("--wiki_path", type=str, required=True, help="Path to the documents.")
    parser.add_argument(
        "--schema_path", required=True, help="The ElasticSearch index schema.",
    )
    parser.add_argument(
        "--strategy",
        choices=["paragraph", "window", "sentence"],
        type=str.lower,
        help="Text splitting strategy.",
    )
    parser.add_argument("-w", "--window", default=0, help="Window size", type=int)
    parser.add_argument("-s", "--stride", default=0, help="Stride size", type=int)
    parser.add_argument("-b", "--bulk", default=10000, help="Bulk size", type=int)

    args = parser.parse_args()

    with open(args.schema_path, "r") as f:
        schema = json.load(f)
    es_client = Elasticsearch(args.host, port=args.port)
    create_index(es_client, args.index, schema)
    es_client.transport.close()

    fill_index(
        args.wiki_path,
        args.host,
        args.port,
        args.index,
        args.strategy,
        args.window,
        args.stride,
        args.bulk,
    )


if __name__ == "__main__":
    main()
