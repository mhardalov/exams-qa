#!/usr/bin/env bash

set -x

# Fail if any error occurs
set -e


# ElasticSearch configs
ES_HOST="localhost"
ES_PORT=9200
ES_BULK=10000

WIKI_DIR="wiki"
SCHEMAS_DIR="configs/elastic_indecies"

LANGS=(ar hr it lt mk pl pt sq sr tr vi bg hu fr de es)

for lang in "${LANGS[@]}"
do
    WIKI_NAME="${lang}wiki"
    WIKI_PATH="${WIKI_DIR}/${WIKI_NAME}"

    if [[ ! -d  ${WIKI_NAME} ]]; then
        python fill_elastic.py ${WIKI_NAME} ${ES_HOST} \
        --port ${ES_PORT} \
        --wiki_path ${WIKI_PATH} \
        --schema_path "${SCHEMAS_DIR}/${WIKI_NAME}.json" \
        --strategy sentence \
        --bulk ${ES_BULK}
    fi
done
