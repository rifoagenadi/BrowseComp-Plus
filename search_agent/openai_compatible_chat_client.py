import argparse
import csv
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

from rich import print

from openai import OpenAI

import sys as _sys
_sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from searcher.searchers import SearcherType
from glm_zai_client import SearchToolHandler, _persist_response
from prompts import format_query


def _split_reasoning_and_content(
    content: str | None,
) -> tuple[str | None, str | None]:
    if not content:
        return None, None

    cleaned = content.replace("<|im_end|>", "").strip()
    if not cleaned:
        return None, None

    reasoning_text = None
    answer_text = cleaned

    if "<think>" in cleaned and "</think>" in cleaned:
        prefix, remainder = cleaned.split("<think>", 1)
        think_body, suffix = remainder.split("</think>", 1)
        extracted = think_body.strip()
        if extracted:
            reasoning_text = extracted
        answer_text = f"{prefix}{suffix}".strip()
    elif "</think>" in cleaned:
        prefix, suffix = cleaned.split("</think>", 1)
        extracted = prefix.strip()
        if extracted:
            reasoning_text = extracted
        answer_text = suffix.strip()

    if answer_text and answer_text.strip():
        answer_text = answer_text.strip()
    else:
        answer_text = None

    return reasoning_text, answer_text


def _normalize_tool_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = str(name).strip()
    if normalized.startswith("functions."):
        normalized = normalized.split(".", 1)[1]
    return normalized or None


def _coerce_tool_arguments(arguments: Any) -> dict[str, Any] | None:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _extract_textual_tool_calls(
    content: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    if not content:
        return [], content

    extracted_calls: list[dict[str, Any]] = []
    cleaned_content = content

    for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL):
        payload_text = match.group(1).strip()
        try:
            payload = json.loads(payload_text)
        except Exception:
            continue

        tool_name = _normalize_tool_name(payload.get("name"))
        tool_args = _coerce_tool_arguments(payload.get("arguments"))
        if not tool_name or tool_args is None:
            continue

        extracted_calls.append(
            {
                "id": f"synthetic-tool-{uuid.uuid4().hex}",
                "name": tool_name,
                "arguments": tool_args,
                "arguments_json": json.dumps(tool_args),
            }
        )

    if extracted_calls:
        cleaned_content = re.sub(
            r"<tool_call>\s*\{.*?\}\s*</tool_call>",
            "",
            cleaned_content,
            flags=re.DOTALL,
        )
        cleaned_content = re.sub(
            r"<tool_response>.*?</tool_response>",
            "",
            cleaned_content,
            flags=re.DOTALL,
        )
        cleaned_content = cleaned_content.strip() or None
        return extracted_calls, cleaned_content

    stripped = content.strip()
    try:
        payload = json.loads(stripped)
    except Exception:
        return [], content

    if not isinstance(payload, dict):
        return [], content

    tool_name = _normalize_tool_name(payload.get("name"))
    tool_args = _coerce_tool_arguments(payload.get("arguments"))
    if not tool_name or tool_args is None:
        return [], content

    extracted_calls.append(
        {
            "id": f"synthetic-tool-{uuid.uuid4().hex}",
            "name": tool_name,
            "arguments": tool_args,
            "arguments_json": json.dumps(tool_args),
        }
    )
    return extracted_calls, None


def _safe_trace_name(value: str | None) -> str:
    if not value:
        return "single"
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return normalized or "single"


def _safe_json_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(k): _safe_json_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_safe_json_value(item) for item in value]
    return str(value)


def _exception_payload(error: Exception) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": type(error).__name__,
        "message": str(error),
    }

    for attr in ("status_code", "code", "param", "request_id"):
        value = getattr(error, attr, None)
        if value is not None:
            payload[attr] = _safe_json_value(value)

    body = getattr(error, "body", None)
    if body is not None:
        payload["body"] = _safe_json_value(body)

    response = getattr(error, "response", None)
    if response is not None:
        status_code = getattr(response, "status_code", None)
        if status_code is not None:
            payload["status_code"] = status_code
        url = getattr(response, "url", None)
        if url:
            payload["url"] = str(url)
        response_text = getattr(response, "text", "")
        if isinstance(response_text, str) and response_text.strip():
            payload["response_text"] = response_text.strip()[:1000]

    return payload


def _persist_failed_query(out_dir: str, *, model: str, query_id: str, args, error: Exception) -> None:
    error_payload = _exception_payload(error)
    normalized_results = [
        {
            "type": "output_text",
            "tool_name": None,
            "arguments": None,
            "output": (
                "Explanation: The query failed due to a harness or model API error.\n"
                "Exact Answer: None\n"
                "Confidence: 0%"
            ),
            "error": error_payload,
        }
    ]
    _persist_response(
        out_dir,
        model=model,
        query_id=query_id,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        normalized_results=normalized_results,
        cumulative_usage={
            "prompt_tokens": 0,
            "prompt_tokens_cached": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "reasoning_tokens": 0,
        },
        finish_reason="failed",
    )


def _dump_raw_iteration(
    dump_root: str | None,
    trace_name: str | None,
    iteration: int,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    parsed_payload: dict[str, Any],
):
    if not dump_root:
        return

    trace_dir = Path(dump_root) / _safe_trace_name(trace_name)
    trace_dir.mkdir(parents=True, exist_ok=True)
    dump_path = trace_dir / f"iter_{iteration:03d}.json"

    with dump_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "iteration": iteration,
                "request": request_payload,
                "response": response_payload,
                "parsed": parsed_payload,
            },
            f,
            indent=2,
            default=str,
        )


def _reduce_max_tokens_for_context_error(
    error: Exception,
    requested_max_tokens: int,
    *,
    safety_margin: int = 32,
) -> int | None:
    message = str(error)
    match = re.search(
        r"maximum context length is\s+(\d+)\s+tokens\s+and your request has\s+(\d+)\s+input tokens",
        message,
        re.IGNORECASE,
    )
    if not match:
        return None

    max_context_tokens = int(match.group(1))
    input_tokens = int(match.group(2))
    allowed_max_tokens = max_context_tokens - input_tokens - safety_margin
    if allowed_max_tokens < 1:
        return None

    adjusted_max_tokens = min(requested_max_tokens, allowed_max_tokens)
    if adjusted_max_tokens >= requested_max_tokens:
        return None
    return adjusted_max_tokens


def run_conversation_with_tools(
    client: OpenAI,
    *,
    query: str,
    model: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    query_template: str | None = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_iterations: int = 100,
    raw_dump_dir: str | None = None,
    raw_trace_name: str | None = None,
):
    tools = tool_handler.get_tool_definitions()

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    formatted_query = format_query(query, query_template)
    messages.append({"role": "user", "content": formatted_query})

    cumulative_usage = {
        "prompt_tokens": 0,
        "prompt_tokens_cached": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
    }

    normalized_results: list[dict[str, Any]] = []

    finish_reason: Optional[str] = None
    global_max_tokens = max_tokens

    for iteration_index in range(1, max_iterations + 1):
        remaining_tokens = global_max_tokens - cumulative_usage["completion_tokens"]
        if remaining_tokens <= 0:
            print(f"Warning: Reached global max_tokens output budget ({global_max_tokens})")
            break

        request_max_tokens = remaining_tokens
        while True:
            create_kwargs = {
                "model": model,
                "messages": messages,
                "tools": tools,
                "max_tokens": request_max_tokens,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature
            if top_p is not None:
                create_kwargs["top_p"] = top_p

            try:
                completion = client.chat.completions.create(**create_kwargs)
                break
            except Exception as exc:
                adjusted_max_tokens = _reduce_max_tokens_for_context_error(
                    exc,
                    request_max_tokens,
                )
                if adjusted_max_tokens is None:
                    raise

                print(
                    "Warning: Reducing max_tokens from "
                    f"{request_max_tokens} to {adjusted_max_tokens} for iteration "
                    f"{iteration_index} due to context budget."
                )
                request_max_tokens = adjusted_max_tokens

        completion_dict = completion.model_dump(mode="python")

        choice = completion.choices[0]
        finish_reason = choice.finish_reason

        usage = getattr(completion, "usage", None)
        if usage is not None:
            cumulative_usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0)
            cumulative_usage["completion_tokens"] += getattr(usage, "completion_tokens", 0)
            cumulative_usage["total_tokens"] += getattr(usage, "total_tokens", 0)

            comp_details = getattr(usage, "completion_tokens_details", None)
            if comp_details is not None:
                cumulative_usage["reasoning_tokens"] += getattr(comp_details, "reasoning_tokens", 0) or 0

            cached_this = 0
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            if prompt_details is not None and getattr(prompt_details, "cached_tokens", None) is not None:
                try:
                    cached_this = int(getattr(prompt_details, "cached_tokens", 0) or 0)
                except Exception:
                    cached_this = 0
            else:
                try:
                    cached_this = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
                except Exception:
                    cached_this = 0

            cumulative_usage["prompt_tokens_cached"] += cached_this

        assistant_msg = choice.message.model_dump()

        reasoning_output = assistant_msg.pop("reasoning_content", None)
        content_reasoning, cleaned_content = _split_reasoning_and_content(
            assistant_msg.get("content")
        )
        textual_tool_calls, cleaned_content = _extract_textual_tool_calls(cleaned_content)

        if isinstance(reasoning_output, str):
            reasoning_output = reasoning_output.strip() or None

        if not reasoning_output:
            reasoning_output = content_reasoning

        if reasoning_output:
            normalized_results.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": [reasoning_output] if isinstance(reasoning_output, str) else reasoning_output,
            })

        assistant_msg["content"] = cleaned_content
        effective_tool_calls: list[dict[str, Any]] = []

        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        for tool_call in raw_tool_calls:
            effective_tool_calls.append(
                {
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments),
                    "arguments_json": tool_call.function.arguments,
                }
            )

        if not effective_tool_calls and textual_tool_calls:
            effective_tool_calls = textual_tool_calls
            assistant_msg["tool_calls"] = [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["name"],
                        "arguments": call["arguments_json"],
                    },
                }
                for call in effective_tool_calls
            ]

        _dump_raw_iteration(
            raw_dump_dir,
            raw_trace_name,
            iteration_index,
            request_payload=create_kwargs,
            response_payload=completion_dict,
            parsed_payload={
                "finish_reason": finish_reason,
                "assistant_message": assistant_msg,
                "reasoning_output": reasoning_output,
                "content_reasoning": content_reasoning,
                "cleaned_content": cleaned_content,
                "textual_tool_calls": textual_tool_calls,
                "effective_tool_calls": effective_tool_calls,
            },
        )

        messages.append(assistant_msg)

        if assistant_msg["content"] is not None and assistant_msg["content"].strip():
            normalized_results.append({
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": assistant_msg["content"],
            })

        if not effective_tool_calls:
            break

        for tool_call in effective_tool_calls:
            tname = tool_call["name"]
            targs = tool_call["arguments"]

            normalized_results.append({
                "type": "tool_call",
                "tool_name": tname,
                "arguments": tool_call["arguments_json"],
                "output": None,
            })

            try:
                output = tool_handler.execute_tool(tname, targs)
                normalized_results[-1]["output"] = output

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tname,
                    "content": output,
                })
            except Exception as e:
                error_msg = f"Error executing {tname}: {str(e)}"
                print(error_msg)
                normalized_results[-1]["output"] = error_msg

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tname,
                    "content": error_msg,
                })

    if finish_reason is None:
        print(f"Warning: Conversation hit max iterations ({max_iterations}) without final response")

    return normalized_results, cumulative_usage, finish_reason


def _process_tsv_dataset(tsv_path: str, client: OpenAI, args, tool_handler: SearchToolHandler):
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved and meta.get("status") == "completed":
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    from tqdm import tqdm
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} ..."
    )

    failures: list[tuple[str, Exception]] = []
    completed_lock = threading.Lock()
    completed_count = [0]
    dump_limit = args.dump_raw_limit
    dumped_counter = [0]

    def _should_dump_raw() -> bool:
        if not args.dump_raw_responses:
            return False
        if dump_limit is None:
            return True
        with completed_lock:
            if dumped_counter[0] >= dump_limit:
                return False
            dumped_counter[0] += 1
            return True

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        raw_dump_dir = None
        raw_trace_name = None
        if _should_dump_raw():
            raw_dump_dir = str(Path(args.output_dir) / "raw_responses")
            raw_trace_name = f"query_{qid}"

        normalized_results, cumulative_usage, finish_reason = run_conversation_with_tools(
            client,
            query=qtext,
            model=args.model,
            max_tokens=args.max_tokens,
            tool_handler=tool_handler,
            system_prompt=args.system,
            query_template=args.query_template,
            temperature=args.temperature,
            top_p=args.top_p,
            max_iterations=args.max_iterations,
            raw_dump_dir=raw_dump_dir,
            raw_trace_name=raw_trace_name,
        )

        with completed_lock:
            completed_count[0] += 1
            if pbar:
                pbar.set_postfix(completed=completed_count[0])

        _persist_response(
            args.output_dir,
            model=args.model,
            query_id=qid,
            system_prompt=args.system,
            max_tokens=args.max_tokens,
            normalized_results=normalized_results,
            cumulative_usage=cumulative_usage,
            finish_reason=finish_reason,
        )

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                try:
                    _handle_single_query(qid, qtext, pbar)
                except Exception as exc:
                    print(f"[Error] Query id={qid} failed: {exc}")
                    _persist_failed_query(args.output_dir, model=args.model, query_id=qid, args=args, error=exc)
                    failures.append((qid, exc))
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            future_to_qid = {
                executor.submit(_handle_single_query, qid, qtext, pbar): qid
                for qid, qtext in remaining
            }
            for future in as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[Error] Query id={qid} failed: {exc}")
                    _persist_failed_query(args.output_dir, model=args.model, query_id=qid, args=args, error=exc)
                    failures.append((qid, exc))
                finally:
                    pbar.update(1)

    if failures:
        failed_qids = ", ".join(qid for qid, _ in failures[:20])
        suffix = " ..." if len(failures) > 20 else ""
        message = f"{len(failures)} query runs failed. Query ids: {failed_qids}{suffix}"
        if args.fail_on_query_error:
            raise RuntimeError(message)
        print(f"[Warning] {message}. Failed query records were saved with status='failed'.")


def main():
    parser = argparse.ArgumentParser(
        description="Call an OpenAI-compatible chat completions API with native function calling and local search."
    )
    parser.add_argument("--query", default="topics-qrels/queries.tsv", help="User query text or path to TSV. Wrap in quotes if contains spaces.")
    parser.add_argument("--model", default="glm-4.6", help="Model name (default: %(default)s)")
    parser.add_argument("--max_tokens", type=int, default=20000, help="Max tokens to generate (default: %(default)s)")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    parser.add_argument("--output-dir", default="runs/bm25/openai_compatible_chat", help="Directory to store logs (default: %(default)s)")
    parser.add_argument("--query-template", choices=["QUERY_TEMPLATE", "QUERY_TEMPLATE_NO_GET_DOCUMENT", "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION"], default="QUERY_TEMPLATE_NO_GET_DOCUMENT", help="Specify the query template to use")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for the model (default: use model defaults)")
    parser.add_argument("--top_p", type=float, default=None, help="Top P for the model (default: use model defaults)")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )
    parser.add_argument(
        "--api-base",
        default="https://api.z.ai/api/paas/v4/",
        help="Base URL for the OpenAI-compatible API (default: %(default)s)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key to use for the OpenAI-compatible API. If omitted, read from --api-key-env.",
    )
    parser.add_argument(
        "--api-key-env",
        default="ZAI_API_KEY",
        help="Environment variable that stores the API key when --api-key is omitted (default: %(default)s)",
    )
    parser.add_argument(
        "--dump-raw-responses",
        action="store_true",
        help="Save raw request/response payloads for each iteration under OUTPUT_DIR/raw_responses.",
    )
    parser.add_argument(
        "--dump-raw-limit",
        type=int,
        default=None,
        help="Maximum number of dataset queries to dump when --dump-raw-responses is enabled. Default: dump all.",
    )
    parser.add_argument(
        "--fail-on-query-error",
        action="store_true",
        help="Exit non-zero if any dataset query fails. By default, failed queries are saved with status='failed' and the run continues.",
    )

    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )

    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="If set, register both the search tool and the get_document tool.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models. If not provided, will use environment variables or CLI login.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets. If not provided, will use environment variables or default.",
    )

    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ["HF_HOME"] = args.hf_home

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"API key is missing. Pass --api-key or set {args.api_key_env} in the environment."
        )

    client = OpenAI(api_key=api_key, base_url=args.api_base)

    searcher = searcher_class(args)
    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    tools_str = ", ".join(tools_registered)

    print(
        f"Search agent started with {searcher.search_type} search (snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {tools_str}")

    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler)
                    return
            except OSError:
                pass

    print(f"Sending request to {args.api_base} with function calling...")
    normalized_results, cumulative_usage, finish_reason = run_conversation_with_tools(
        client,
        query=args.query,
        model=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        query_template=args.query_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
        raw_dump_dir=str(Path(args.output_dir) / "raw_responses") if args.dump_raw_responses else None,
        raw_trace_name="single",
    )

    _persist_response(
        args.output_dir,
        model=args.model,
        query_id=None,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        normalized_results=normalized_results,
        cumulative_usage=cumulative_usage,
        finish_reason=finish_reason,
    )

    final_texts = [item["output"] for item in normalized_results if item.get("type") == "output_text"]
    if final_texts:
        print(final_texts[-1])


if __name__ == "__main__":
    main()
