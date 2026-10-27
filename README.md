# The-Ai

## WorkBot Micro-Specialist (free, local "muscle" model)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AaronGrace978/The-Ai/blob/main/WorkBot_Micro_Specialist_Finetune.ipynb)

[`WorkBot_Micro_Specialist_Finetune.ipynb`](WorkBot_Micro_Specialist_Finetune.ipynb) fine-tunes Qwen2.5-Coder 1.5B or 3B with Unsloth (4-bit QLoRA) into a fast, single-purpose execution model:

- **Tool calling**: answers with `<tool_call>` JSON straight away, no chat preamble.
- **Browser actions**: turns a task plus a list of page elements into one `browser_act` call (`CLICK` / `TYPE` / `SELECT` on an element id) for a Tauri/Rust layer to execute.
- **Knowing when not to call**: asks for missing arguments or declines instead of inventing calls.

Training data: Hermes function calling, ToolACE, xLAM-60k (optional, gated), When2Call, Mind2Web browser trajectories, a small Magicoder slice, and your own WorkBot trajectories (optional JSONL).

The notebook evaluates the model on held-out data before and after training, then exports a GGUF file plus an Ollama `Modelfile`. You can run the result offline on an RTX 5060 Ti (or any GPU with a few GB of VRAM) and call it through Ollama's native or OpenAI-compatible API.

| Preset | Base | Hardware |
|---|---|---|
| `colab_free` | Qwen2.5-Coder-1.5B-Instruct | free Colab T4 (about 1 to 1.5 h of training) |
| `quality` | Qwen2.5-Coder-3B-Instruct | Colab L4/A100 or a local 16 GB GPU (overnight) |

This is a specialist for high-volume execution steps, not a replacement for a frontier model's planning. Pair it with a larger "mind" model that handles strategy.
