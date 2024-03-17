# Web UI Operating instructions

## 1. Initialization

Input config file path and choose [milvus|pipeline] then click the first `Initialization`，first time downloading the model may be slow, but after completion, it will display `CLI Initialization completed`

## 2. Build index

Choose `Build index`then input the file to construct corpus, example `./data/bitcoin.pdf`, bitcoin.pdf will be input. You can see the progress of embedding generation in the consol. Note that if the corpus is large, like `./data/` will input all files in this path, may take a longer time. It is recommended to use the Zilliz Cloud Pipelines for large-scale corpora.

## 3. Input question

Choose `Ask` or `Ask + Return to retrieve content`，then input your question and click `submit` to get the answer.