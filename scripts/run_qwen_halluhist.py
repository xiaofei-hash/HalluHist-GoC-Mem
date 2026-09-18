"""Resumable HalluHist inference with Qwen2.5-VL and the released GoC-Mem pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from pathlib import Path

from goc_mem.core import score_pairs
from goc_mem.pipeline import run_sample


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raw = raw[: raw.rfind(b"\n") + 1]
        path.write_bytes(raw)
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class QwenBackend:
    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)

    def generate_text(
        self,
        prompt: str,
        image_path: str | None = None,
        max_new_tokens: int = 128,
    ) -> str:
        from qwen_vl_utils import process_vision_info

        content = []
        if image_path is not None:
            content.append({"type": "image", "image": str(image_path)})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        rendered = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if image_path is None:
            inputs = self.processor(
                text=[rendered], padding=True, return_tensors="pt"
            ).to(self.model.device)
        else:
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[rendered],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        generated = generated[:, inputs.input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path("data/evaluation_inputs.jsonl"))
    parser.add_argument("--labels", type=Path, default=Path("data/evaluation_labels.jsonl"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tau-c", type=float, default=0.6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = read_jsonl(args.inputs)
    labels = read_jsonl(args.labels)
    selected = inputs[args.start :]
    if args.limit is not None:
        selected = selected[: args.limit]
    for row in selected:
        candidate = args.image_root / Path(row["image_path"]).name
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        row["image_path"] = str(candidate)

    protocol = {
        "dataset": "HalluHist",
        "inputs": str(args.inputs.resolve()),
        "inputs_sha256": sha256(args.inputs),
        "labels": str(args.labels.resolve()),
        "labels_sha256": sha256(args.labels),
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256(args.model / "config.json"),
        "tau_c": args.tau_c,
        "cascade": True,
        "extraction_max_new_tokens": 1024,
        "verification_max_new_tokens_per_call": 128,
        "answer_max_new_tokens": 32,
        "do_sample": False,
        "start": args.start,
        "limit": args.limit,
        "selected_samples": len(selected),
        "source_commit": git_commit(root),
        "python": platform.python_version(),
    }
    write_json(args.output / "protocol.json", protocol)

    controller_path = args.output / "controller.jsonl"
    predictions_path = args.output / "predictions.jsonl"
    complete = {row["sample_id"] for row in read_jsonl(controller_path)}
    prediction_rows = read_jsonl(predictions_path)
    if {row["sample_id"] for row in prediction_rows} != complete:
        raise RuntimeError("Controller and prediction checkpoints disagree")

    backend = QwenBackend(args.model)
    started = time.time()
    for index, sample in enumerate(selected, 1):
        sample_id = sample["sample_id"]
        if sample_id in complete:
            continue
        sample_started = time.time()
        for attempt in range(2):
            try:
                result = run_sample(sample, backend, tau_c=args.tau_c, cascade=True)
                break
            except Exception as exc:
                if attempt == 1:
                    append_jsonl(args.output / "failures.jsonl", {
                        "sample_id": sample_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    })
                    raise
                backend.torch.cuda.empty_cache()
        result["seconds"] = time.time() - sample_started
        append_jsonl(controller_path, result)
        append_jsonl(predictions_path, {"sample_id": sample_id, "answer": result["answer"]})
        complete.add(sample_id)
        elapsed = time.time() - started
        done = len(complete)
        remaining = max(0, len(selected) - done)
        status = {
            "state": "running",
            "completed": done,
            "total": len(selected),
            "last_sample_id": sample_id,
            "last_seconds": result["seconds"],
            "elapsed_seconds_this_run": elapsed,
            "estimated_remaining_seconds": (elapsed / max(1, index)) * remaining,
            "updated_unix": time.time(),
        }
        write_json(args.output / "status.json", status)
        print(json.dumps(status), flush=True)

    predictions = read_jsonl(predictions_path)
    if len(selected) == len(inputs) and len(predictions) == len(labels):
        metrics = score_pairs(labels, predictions)
        write_json(args.output / "metrics.json", metrics)
    else:
        metrics = {"note": "Partial run; full paired metrics are not computed."}
    write_json(args.output / "status.json", {
        "state": "complete",
        "completed": len(predictions),
        "total": len(selected),
        "metrics": metrics,
        "updated_unix": time.time(),
    })
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
