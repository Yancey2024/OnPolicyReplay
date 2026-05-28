# On-Policy Replay (OPR)

A minimal reproduction of the EMNLP paper
**"On-Policy Replay: Treating Self-Generated Rollouts as a Replay Source Mitigates Forgetting in Continual Supervised Fine-Tuning"**.

OPR is a simple recipe for continual SFT: at the end of each task, the most recent checkpoint rolls out responses on historical prompts, the top-scoring (prompt, response) pairs form a tiny replay buffer, and the next task is trained on `new_data + buffer` with plain cross-entropy. No teacher, no auxiliary loss.

This repo implements the two scorer variants from the paper:
- **OPR-RU**: rule-based scorer (task metric).
- **OPR-SC**: self-confidence scorer (length-normalized log-prob).

The benchmark is **TRACE** (8 tasks): `C-STANCE → FOMC → MeetingBank → Py150 → ScienceQA → NumGLUE-cm → NumGLUE-ds → 20Minuten`.

## Requirements

- `ms-swift` (for SFT)
- `vllm`
- `transformers`, `rouge_score`, `fuzzywuzzy`, `evaluate`
- 8x GPUs (the training/eval scripts assume `tensor_parallel_size=8`)

## Data

Download the 8 TRACE datasets and place them under `data/` as below:

```
data/
├── C-STANCE/{train,test}.jsonl
├── FOMC/{train,test}.jsonl
├── MeetingBank/{train,test}.jsonl
├── Py150/{train,test}.jsonl
├── ScienceQA/{train,test}.jsonl
├── NumGLUE-cm/{train,test}.jsonl
├── NumGLUE-ds/{train,test}.jsonl
└── 20Minuten/{train,test}.jsonl
```

Each line is a JSON object with two fields: `{"prompt": ..., "answer": ...}`.

> Download link: https://drive.google.com/file/d/1S0SmU0WEw5okW_XvP2Ns0URflNzZq6sV/view?usp=drive_link

## Model

Download a base instruct model (default: `Qwen2.5-7B-Instruct`) and put it under `model/Qwen2.5-7B-Instruct/`.

## Run

```bash
cd TrueCode
python main.py \
    --model-path model/Qwen2.5-7B-Instruct \
    --output-dir output/qwen_opr_ru \
    --rho 0.01
```

- `--rho` is the replay ratio (buffer size = `rho * 5000`). The paper reports results at 1% and 10%.
- The loop trains task 1..8 in order. After each task it evaluates on all seen tasks and (except for the last task) regenerates the buffer with the new checkpoint.

To switch from **OPR-RU** (default) to **OPR-SC**, change the script name in `OPRTrainer.py::generate` from `script/generate_opr_ru.py` to `script/generate_opr_sc.py`.

## Outputs

- Checkpoints: `output/qwen_opr_ru/<run-id>/checkpoint-*`
- Per-task scores after each stage: `<checkpoint-dir>/../conclusion.json`
- Generated replay buffers: `data/<next-task>/buffer.jsonl`

## Files

| File | Purpose |
|---|---|
| `main.py` | Entry point. |
| `OPRTrainer.py` | Train → eval → generate-buffer loop. |
| `script/train.sh` | `swift sft` full-parameter training (DeepSpeed ZeRO-2). |
| `script/eval.py` | Multi-task evaluation on TRACE. |
| `script/generate_opr_ru.py` | Build buffer with rule-based scores. |
| `script/generate_opr_sc.py` | Build buffer with self-confidence scores. |
