<p align="center">
  <img src="https://img.shields.io/badge/vLLM-0.25.1-4f8ef7?style=flat-square" alt="vLLM">
  <img src="https://img.shields.io/badge/model-Gemma--4--31B--IT--NVFP4-1a73e8?style=flat-square" alt="Model">
  <img src="https://img.shields.io/badge/quantization-NVFP4-34a853?style=flat-square" alt="Quantization">
  <img src="https://img.shields.io/badge/license-Gemma-ea4335?style=flat-square" alt="License">
</p>

# Gemma 4 31B IT — NVFP4

Serve **Google Gemma 4 31B IT** in 4-bit NVFP4 quantization via vLLM with an OpenAI-compatible endpoint, Multi-Token Prediction speculative decoding, tool calling, and thinking/reasoning support.

Automatically downloads and caches both models from Hugging Face — just set your token and run.

---

## Models

| Role | Model | Size | Precision |
|------|-------|------|-----------|
| 🎯 **Target** | [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) | ~31B | NVFP4 (ModelOpt) |
| ⚡ **Draft** | [`google/gemma-4-31B-it-assistant`](https://huggingface.co/google/gemma-4-31B-it-assistant) | ~0.5B | BF16 |

The draft model runs 4 lightweight decoder layers with Q-only attention, sharing KV cache with the target. It produces up to **4 draft tokens per step** for speculative decoding — accelerating generation without sacrificing output quality.

---

## Quick Start

### Prerequisites

- **Docker** with NVIDIA Container Toolkit (`nvidia-ctk` completed)
- **NVIDIA GPU** with ≥22 GB VRAM (for 256k context at `gpu-memory-utilization 0.7`)
- **Hugging Face token** with access to both gated models:
  - https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
  - https://huggingface.co/google/gemma-4-31B-it-assistant

### Start the server

```bash
HF_TOKEN=hf_your_token_here ./start.sh
```

The script downloads both models on first run (cached in `~/.cache/huggingface`), launches the vLLM container, and waits for it to be ready.

### Use the API

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Gemma-4-31B-IT-NVFP4",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Stop the server

```bash
./stop.sh
```

---

## Features

### 🧠 Thinking / Reasoning

Enabled by default. The model outputs its reasoning in a `<channel>thought` block before the final response:

```
<|turn|>model
<|channel|>thought
The user is asking about...
<channel|>Here's the answer...
<turn|>
```

Disable per-request by passing `chat_template_kwargs: {"enable_thinking": false}` in your API call.

### 🛠️ Tool Calling

Native tool calling via the `gemma4` parser. Everything is pre-configured — just pass `tools` in your request:

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Gemma-4-31B-IT-NVFP4",
    "messages": [{"role": "user", "content": "What'\''s the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": { "type": "string", "description": "City name" }
          },
          "required": ["location"]
        }
      }
    }]
  }'
```

### 🖼️ Multi-modal (Image + Video)

The model supports images and video through a shared vision encoder. Audio has a token ID defined but the NVFP4 quantized weights do not include an audio encoder (`audio_config: null`).

**Image example:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        {"type": "text", "text": "What's in this image?"}
      ]
    }
  ]
}
```

**Video example:**
```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}},
        {"type": "text", "text": "Summarize this video"}
      ]
    }
  ]
}
```

Limits: up to **4 images** or **1 video** (or **1 audio** if supported) per prompt.

### ⚡ MTP Speculative Decoding

Multi-Token Prediction generates 4 draft tokens per step using the lightweight assistant model. Most effective for:
- **Batched** workloads
- **Long-context** generation
- **System-prompt-heavy** applications (prefix caching also active)

---

## Concurrency & Performance

### Limits

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--max-num-seqs` | `8` | Hard cap on concurrent sequences |
| `--max-num-batched-tokens` | `8192` | Total tokens across all sequences in a single batch |
| `--gpu-memory-utilization` | `0.70` | VRAM budget (~22 GB on a 32 GB GPU) |

### Realistic throughput

| Workload | Typical concurrent requests |
|----------|---------------------------|
| Short queries (≤1K tokens each) | **8** (hits `max-num-seqs` cap) |
| Coding / tool calling (500–2K tokens) | **4–8** |
| Long context (32K+ tokens each) | **1–3** (VRAM-bound) |

The `max-num-batched-tokens` of 8192 is the practical bottleneck for short concurrent requests. If 8 users each send a 2K-token prompt, the total (16K) exceeds the batch budget — vLLM drains and refills dynamically.

### Tuning for higher concurrency

To increase throughput at the cost of higher VRAM usage:

```bash
# In start.sh, increase these values:
--max-num-seqs 16
--max-num-batched-tokens 16384
--gpu-memory-utilization 0.85
```

Monitor with `nvidia-smi` to ensure you don't OOM.

---

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | — | Hugging Face token for gated models |
| `PORT` | `8888` | Server port |
| `HOST` | `0.0.0.0` | Bind address |

### Key vLLM flags

| Flag | Value | Notes |
|------|-------|-------|
| `--quantization` | `modelopt` | NVFP4 format from NVIDIA ModelOpt |
| `--tensor-parallel-size` | `1` | Bump for multi-GPU |
| `--gpu-memory-utilization` | `0.70` | Tune to your VRAM budget |
| `--max-model-len` | `262144` | 256k context window |
| `--kv-cache-dtype` | `fp8` | Reduces KV cache memory by ~50% |
| `--limit-mm-per-prompt` | `{"image":4,"video":1,"audio":1}` | Max images, video, or audio per request |
| `--chat-template` | `./chat_template.jinja` | Gemma 4 canonical template |
| `--reasoning-parser` | `gemma4` | Parses thinking blocks |
| `--tool-call-parser` | `gemma4` | Native tool call format |
| `--attention-backend` | `triton_attn` | Triton-based flash attention |
| `--load-format` | `fastsafetensors` | Fast local model loading |
| `--speculative-config` | *(see start.sh)* | MTP with assistant model |
| `--override-generation-config` | `{"temperature":1.0,...}` | Default sampling params |

### Per-request overrides

Clients can override the template kwargs per-request:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false,
    "preserve_thinking": false
  }
}
```

---

## Files

```
.
├── start.sh               # Launch vLLM container (auto-downloads models)
├── stop.sh                # Gracefully stop the container
├── chat_template.jinja    # Gemma 4 canonical chat template (390 lines)
├── README.md              # This file
├── .gitignore             # Ignores runtime artifacts
├── .vllm.log              # Container logs (git-ignored)
├── .vllm.pid              # Container PID (git-ignored)
└── .cache/                # Triton cache (git-ignored)
```

---

## Python Client Examples

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8888/v1",
    api_key="not-needed",
)

response = client.chat.completions.create(
    model="nvidia/Gemma-4-31B-IT-NVFP4",
    messages=[{"role": "user", "content": "Write a quick sort in Python"}],
    temperature=0.2,
)

print(response.choices[0].message.content)
```

### With tool calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code and return stdout",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"}
                },
                "required": ["code"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="nvidia/Gemma-4-31B-IT-NVFP4",
    messages=[{"role": "user", "content": "Calculate 42 * 37"}],
    tools=tools,
)
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Container exits immediately | Out of VRAM | Lower `--gpu-memory-utilization` to `0.5` |
| Download hangs | Missing HF token | Pass `HF_TOKEN=...` |
| Port conflict | Something on :8888 | `PORT=8889 ./start.sh` |
| Slow first start | Downloading ~19 GB | Normal — cached after first run |
| `Unknown vLLM env var` warnings | Build metadata in image | Harmless, can ignore |
| MTP not speeding up | Short single-turn prompts | Most effective with batching / long contexts |

---

## License

The model weights are governed by the [Gemma License](https://www.kaggle.com/models/google/gemma-4/license). This repository's scripts and configuration are provided under the MIT License.

---

## References

- [Gemma 4 technical report](https://goo.gle/Gemma4Report)
- [NVIDIA ModelOpt NVFP4](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [vLLM documentation](https://docs.vllm.ai/en/latest/)
- [Hugging Face: nvidia/Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
