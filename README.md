# Kalypso: Relational LLM Serving

This repository is built on top of vLLM `v0.13.0rc4` with an added semantic
query processor under `semantic_query_processor/vllm/semantic_query_processor`.

The semantic query processor runs inside the vLLM OpenAI-compatible server and
adds semantic operators such as filtering, mapping, joins, aggregation, top-k
ranking, and indexed search over local data files.

## Install

Clone the repository and install it from source:

```bash
cd /home/hojaeson_umass/semantic_query_processor

pip install -U pip setuptools wheel ninja cmake packaging
pip install -r requirements/build.txt
pip install -r requirements/common.txt

pip install -e . --no-build-isolation
```

If you use a virtual environment or conda environment, activate it before
running the commands above.

## Run vLLM

Start the vLLM OpenAI-compatible API server with Llama 3.3 70B:

```bash
cd /home/hojaeson_umass/semantic_query_processor

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8003
```

Equivalent JSON-style args:

```json
{
  "args": [
    "--model",
    "meta-llama/Llama-3.3-70B-Instruct",
    "--tensor-parallel-size",
    "4",
    "--gpu-memory-utilization",
    "0.8",
    "--enable-prefix-caching",
    "--max-model-len",
    "32768",
    "--port",
    "8003"
  ],
  "env": {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0"
  }
}
```


For the 70B setup above, pass `cascade_model` explicitly if you use cascade
operators.

## Run ICP Service

Some benchmark pipelines use ICP/indexed retrieval. Start the ICP service before
running those clients.

For BioDEX, use the default FAISS backend:

```bash
cd /home/hojaeson_umass/semantic_query_processor

python vllm/semantic_query_processor/icp/vector_service.py \
  --host 127.0.0.1 \
  --port 8080
```

For FEVER, use the ColBERT backend:

```bash
cd /home/hojaeson_umass/semantic_query_processor

python vllm/semantic_query_processor/icp/vector_service.py \
  --host 127.0.0.1 \
  --port 8080 \
  --backend colbert
```

## Run Cascade Model

Cascade/proxy filtering should use a separate vLLM endpoint. For example, run a
Llama 8B server on port `8004` and configure benchmark clients with
`cascade_api_base="http://localhost:8004/v1"`:

```bash
cd /home/hojaeson_umass/semantic_query_processor

VLLM_ENABLE_V1_MULTIPROCESSING=0 vllm serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.9 \
  --enable-prefix-caching \
  --max-model-len 32768 \
  --port 8004
```

## Pipeline Examples

The example clients live in the benchmark directory:

```bash
cd /home/hojaeson_umass/semantic_query_processor/vllm/semantic_query_processor/benchmark
```

Each client sends a request to the semantic query endpoint. By default, the
clients use `meta-llama/Llama-3.3-70B-Instruct` and port `8003`.

Small benchmark datasets are bundled under:

```bash
/home/hojaeson_umass/semantic_query_processor/vllm/semantic_query_processor/benchmark/sample_data
```

Included sample data contains 10 records per dataset:

- `fever_claims_sample_1000_data.csv`
- `MEDEC-TrainingSet-1000.csv`
- `articles_500/`
- `reactions/`
- `contract-nli/`

Full benchmark datasets are available as zip files on Google Drive:

`https://drive.google.com/drive/u/0/folders/1N2UvdBGyHPgq5FjdA_FDtCegItCdC8pd`

Run the FEVER Factool map, indexed search, and filter pipeline:

```bash
python client_fever_factool_map_search_filter_op.py
```

Run the FEVER Factool map, ICP, and filter pipeline:

```bash
python client_fever_factool_map_icp_filter.py
```

Run the MEDEC filter and two-map pipeline:

```bash
python client_medec_filter_map_map.py
```

Run the BioDEX map and ICP pipeline:

```bash
python client_biodex_map_icp.py
```

Run the Contract NLI filter, join, and map pipeline:

```bash
python client_contract_nli_filter_join_map.py
```
