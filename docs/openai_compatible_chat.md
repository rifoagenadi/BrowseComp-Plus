# Evaluating Models On OpenAI-Compatible Chat Completions APIs

First, please make sure that you have followed the main README.md to download the decrypted dataset, setup your environment, and downloaded the indexes you need.

Use `search_agent/openai_compatible_chat_client.py` when your inference endpoint exposes an OpenAI-compatible `chat/completions` API with tool calling support.

You can authenticate by passing `--api-key` directly or by setting the environment variable named by `--api-key-env` (defaults to `ZAI_API_KEY`).

For example, to target a custom endpoint:
```bash
python search_agent/openai_compatible_chat_client.py \
  --api-base http://<base_url>:<port>/v1/ \
  --api-key 'YOUR_API_KEY' \ #optional
  --model <served_model_name> \
  --output-dir <output_path> \
  --searcher-type bm25 \
  --index-path indexes/bm25/ \
  --num-threads 10
```

This requires the remote server to support OpenAI-style tool calling on the `chat/completions` route. If it only supports plain text completion without tool calls, the benchmark agent will not be able to use the retriever.
