#!/usr/bin/env bash

set -x

# Base path to the Wikipedia dumps
WIKI_BASE_URL="https://ftp.acc.umu.se/mirror/wikimedia.org/dumps/"

WIKI_DIR="wiki/"

#This should be updated with the current dump!
DUMP_DATE="20200301"

LANGS=(ar hr it lt mk pl pt sq sr tr vi bg hu fr de es)

# Get the number of available CPUs
PROC=$(nproc)

mkdir -p $WIKI_DIR
cd $WIKI_DIR

for lang in "${LANGS[@]}"
do
    WIKI_NAME="${lang}wiki"
    WIKI_XML="${WIKI_NAME}-${DUMP_DATE}-pages-meta-current.xml"
    WIKI_XML_ZIP="$WIKI_XML.bz2"


    if [[ ! -d  ${WIKI_NAME} ]]; then
        # Skip existing files
        if [[ ! -f  ${WIKI_XML} ]]; then
            wget -N ${WIKI_BASE_URL}${WIKI_NAME}/${DUMP_DATE}/${WIKI_XML_ZIP}
            pbzip2 -d ${WIKI_XML}.bz2 -p$PROC
        fi

        # Convert the dump into JSON
        python ../WikiExtractor.py --output ${WIKI_NAME} --json ${WIKI_XML} --processes $PROC
    fi
done

