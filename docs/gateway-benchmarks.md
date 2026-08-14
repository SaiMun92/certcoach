# ai-core Gateway Benchmarks

Gateway: `http://127.0.0.1:3030/v1` (SAP Generative AI Hub local proxy)
Measured: 2026-08-09 — streaming TTFT, single short prompt ("Say one word."), `max_tokens=1`

## Time to First Token (ranked)

| Rank | Model                    | TTFT (ms) |
|------|--------------------------|-----------|
| 1    | gpt-5                    | 617       |
| 2    | gpt-5-nano               | 666       |
| 3    | gemini-3.5-flash         | 766       |
| 4    | claude-sonnet-4-6        | 856       |
| 5    | gpt-5.5                  | 847       |
| 6    | gemini-2.5-pro           | 882       |
| 7    | claude-opus-4-8          | 1096      |
| 8    | o3                       | 1109      |
| 9    | gpt-5-mini               | 1140      |
| 10   | qwen3.6-flash            | 1239      |
| 11   | qwen3.6-plus             | 1373      |
| 12   | claude-haiku-4-5-20251001| 1423      |
| 13   | sonar-pro                | 1520      |
| 14   | claude-opus-4-6          | 2086      |

## Notes

- **gpt-5 is the fastest** at ~617ms — counterintuitively faster than gpt-5-nano through this proxy.
- **claude-haiku** is surprisingly slow (1423ms) — proxy routing overhead appears to flatten the usual small-model latency advantage.
- **gemini-3.5-flash** is the fastest non-GPT option at 766ms.
- GPT-5 family (`gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5.5`, `o3`) requires `max_completion_tokens` instead of `max_tokens`.
- `sonar-pro` requires `max_tokens >= 16`.
- Non-chat models excluded: `cohere-rerank-pro`, `text-embedding-3-large`, `gemini-embedding`, `amazon--titan-embed-image`.
