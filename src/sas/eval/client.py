"""Shared async sglang HTTP client for SAS evaluation.

Both eval task types (reasoning, longbench) hit the SAME already-running
sglang server (OpenAI-compatible) instead of loading the model in-process. The
server's continuous batching does the heavy lifting; we just keep it saturated
with concurrent requests.

Two entry points:
  * ``chat_completions`` — POST ``/v1/chat/completions`` with ``messages``; the
    server applies the model's chat_template. Pass ``chat_template_kwargs`` to
    control it (e.g. ``{"enable_thinking": False}`` to suppress Qwen3 thinking).
  * ``completions`` — POST ``/v1/completions`` with a raw ``prompt`` string; the
    server applies NO template (used for LongBench's NO_CHAT_TEMPLATE datasets).

Both take a list of per-request payfloads-in and return results in submission
order (so callers can zip results back against their inputs).
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from tqdm import tqdm

# Marker string returned in place of text when a request exhausts all retries.
REQUEST_FAILED_PREFIX = "[REQUEST_FAILED"


async def _one_request(client, url, payload, sem, max_retries):
    """POST one request; return ``(text, n_completion_tokens)``. Retries on error.

    Works for both the chat and raw-completion endpoints — they differ only in
    where the generated text lives in the response, which we handle by trying
    ``message.content`` first (chat) then ``text`` (completions).
    """
    async with sem:
        last_err = None
        for attempt in range(max_retries):
            try:
                r = await client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                if "message" in choice:                 # chat endpoint
                    text = choice["message"]["content"]
                else:                                    # completions endpoint
                    text = choice["text"]
                n_tok = None
                usage = data.get("usage") or {}
                if "completion_tokens" in usage:
                    n_tok = usage["completion_tokens"]
                return text, n_tok
            except Exception as e:  # noqa: BLE001 - retry any transport/HTTP error
                last_err = e
                await asyncio.sleep(2.0 * (attempt + 1))
        return f"{REQUEST_FAILED_PREFIX}: {type(last_err).__name__}: {last_err}]", 0


async def _dispatch(payloads, *, base_url, endpoint, concurrency, timeout,
                    max_retries, desc):
    """Fire all ``payloads`` concurrently at ``endpoint``; keep submission order.

    Returns a list of ``(text, n_completion_tokens)`` aligned with ``payloads``.
    """
    url = base_url.rstrip("/") + endpoint
    sem = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency + 16,
                          max_keepalive_connections=concurrency + 16)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        with tqdm(total=len(payloads), desc=desc) as pbar:
            async def _tracked(p):
                res = await _one_request(client, url, p, sem, max_retries)
                pbar.update(1)
                return res

            # gather preserves submission order, so results line up with payloads.
            return await asyncio.gather(*(_tracked(p) for p in payloads))


def _build_common(model, temperature, top_p, max_tokens, stop, extra):
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if stop is not None:
        body["stop"] = stop
    if extra:
        body.update(extra)
    return body


async def chat_completions(
    messages_list,
    *,
    base_url,
    model,
    max_tokens,
    temperature=0.6,
    top_p=0.95,
    concurrency=256,
    timeout=3600.0,
    max_retries=3,
    stop=None,
    chat_template_kwargs=None,
    desc="chat",
):
    """Run a batch of chat-completion requests (server applies the chat template).

    ``messages_list`` is a list of ``messages`` arrays (one per request). Returns
    ``[(text, n_tok), ...]`` in the same order.
    """
    extra = {}
    if chat_template_kwargs is not None:
        extra["chat_template_kwargs"] = chat_template_kwargs
    payloads = [
        {**_build_common(model, temperature, top_p, max_tokens, stop, extra),
         "messages": messages}
        for messages in messages_list
    ]
    return await _dispatch(
        payloads, base_url=base_url, endpoint="/v1/chat/completions",
        concurrency=concurrency, timeout=timeout, max_retries=max_retries, desc=desc,
    )


async def completions(
    prompt_list,
    *,
    base_url,
    model,
    max_tokens,
    temperature=0.6,
    top_p=0.95,
    concurrency=256,
    timeout=3600.0,
    max_retries=3,
    stop=None,
    desc="completion",
):
    """Run a batch of raw-completion requests (NO chat template applied).

    ``prompt_list`` is a list of raw prompt strings. Returns ``[(text, n_tok), ...]``
    in the same order.
    """
    payloads = [
        {**_build_common(model, temperature, top_p, max_tokens, stop, None),
         "prompt": prompt}
        for prompt in prompt_list
    ]
    return await _dispatch(
        payloads, base_url=base_url, endpoint="/v1/completions",
        concurrency=concurrency, timeout=timeout, max_retries=max_retries, desc=desc,
    )
