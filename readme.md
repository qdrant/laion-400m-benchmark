# Laion 400M Dataset benchmark

This repository provides scripts for uploading and benchmarking the Laion 400M dataset in Qdrant.


## Dataset

[Laion 400M](https://laion.ai/blog/laion-400-open-dataset/) dataset is a dataset of 400M vectors with 512d CLIP embeddings.


## Hardware

Minimal usable configuration for interactive search is: 64Gb RAM, 8 cores CPU, 1Tb Disk.
With this configuration it would be possible to explore the dataset with under 1 second latency.

Higher performance or addititional payload indexes can be achieved by allocating more CPUs and RAM.

**TIP**: Best porformance is achieved if you run qdrant with [async io](https://qdrant.tech/articles/io_uring) enabled. This option available in cloud as well.


## Upload

To upload the dataset to Qdrant, run the following command:

```bash
export QDRANT_URL="https://xxxx-xxxx.xxxx.cloud.qdrant.io"
export QDRANT_API_KEY="xxxx-xxxx-xxxx-xxxx"

python upload.py
```

This script will download chunks of the LAION dataset one by one and upload them to Qdrant.
Intermediate data is not persisted on disk, so the script doesn't require much disk space on the client side.


## Generating reference data

To obtain ground truth for the dataset, we have used full-scan on the dataset.
See `full_scan.py` for the reference implementation.

It require a lot of time and resources, so we attach already generated reference data `exprected.py`.


## Evaluation

To compare search results and measure search latency, run the following command:

```bash
python eval.py --rescore_limit 1000
```

Higher the `rescore_limit` - more accurate the results, but slower the evaluation.

