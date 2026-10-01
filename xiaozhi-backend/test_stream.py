import asyncio, httpx, json, time

async def test():
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": "qwen3:4b-nothink",
        "messages": [
            {"role": "system", "content": "You are a voice assistant. Reply in one short sentence max 15 words. Vietnamese->Vietnamese."},
            {"role": "user", "content": "xin chao"},
        ],
        "think": False,
        "stream": True,
        "options": {"num_ctx": 4096, "num_predict": 2000, "num_thinking_budget": 0, "temperature": 0.2}
    }
    t0 = time.monotonic()
    content_parts = []
    thinking_seen = False
    think_buf = ""
    first_answer_token_t = None

    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream("POST", url, json=payload) as resp:
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                if chunk.get("done"):
                    dr = chunk.get("done_reason","")
                    print(f"done_reason={dr}")
                    break
                msg = chunk.get("message", {})
                think_tok = msg.get("thinking", "")
                content_tok = msg.get("content", "")
                if think_tok:
                    thinking_seen = True
                if content_tok:
                    think_buf += content_tok
                    lower = think_buf.lower()
                    close = lower.find("</think>")
                    if close >= 0:
                        think_buf = think_buf[close+8:].lstrip("\n")
                        if think_buf:
                            content_parts.append(think_buf)
                            think_buf = ""
                    elif "<think>" not in lower:
                        if first_answer_token_t is None:
                            first_answer_token_t = time.monotonic()
                            print(f"First answer token at: {(first_answer_token_t-t0)*1000:.0f}ms")
                        content_parts.append(think_buf)
                        think_buf = ""

    total = time.monotonic() - t0
    answer = "".join(content_parts).strip()
    print(f"Total time: {total*1000:.0f}ms")
    if first_answer_token_t:
        print(f"Time to first answer token: {(first_answer_token_t-t0)*1000:.0f}ms")
    else:
        print("No answer token found")
    print(f"Answer: [{answer}]")

asyncio.run(test())
