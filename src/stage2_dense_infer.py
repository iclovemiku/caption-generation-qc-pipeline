"""
Stage 2: 本地双 Captioner 密集标注（Qwen3-Omni-Captioner + AF-Next-Captioner）。

职责：对同一音频分别推理，产出 qwen_dense_caption、af_next_dense_caption。

用法:
    python -m src.stage2_dense_infer
    python -m src.stage2_dense_infer --input-folder data/01_raw_audio
    python -m src.stage2_dense_infer --source-jsonl data/02_base_captions/base_captions.jsonl
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODEL_DIRS = {
    "Qwen3-Omni-Captioner": PROJECT_ROOT / "models" / "qwen3-omni-captioner",
    "AF-Next-Captioner": PROJECT_ROOT / "models" / "af-next-captioner",
}

PROMPTS = {
    "Qwen3-Omni-Captioner": "Please generate a detailed caption for this audio.",
    "AF-Next-Captioner": (
        "Generate a detailed caption for the input audio. "
        "In the caption, transcribe all spoken content by all speakers in the audio precisely."
    ),
}

DEFAULT_INPUT_FOLDER = str(PROJECT_ROOT / "data" / "01_raw_audio")
DEFAULT_OUTPUT_FILE = str(PROJECT_ROOT / "data" / "03_dense_captions" / "dense_captions.jsonl")
DEFAULT_SOURCE_JSONL = ""

_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Dense captioning using local Qwen + AF-Next models."
    )
    parser.add_argument(
        "--input-folder",
        default=DEFAULT_INPUT_FOLDER,
        help="Folder containing raw audio files.",
    )
    parser.add_argument(
        "--source-jsonl",
        default=DEFAULT_SOURCE_JSONL,
        help="Optional Stage 1 JSONL manifest (to reuse filename list).",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="Output JSONL path for dense captions.",
    )
    parser.add_argument(
        "--qwen-max-tokens",
        type=int,
        default=1024,
        help="max_new_tokens for Qwen model.",
    )
    parser.add_argument(
        "--af-max-tokens",
        type=int,
        default=1024,
        help="max_new_tokens for AF-Next model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of files to process before clearing GPU cache (for memory efficiency).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional limit for quick pilot runs.",
    )
    parser.add_argument(
        "--skip-qwen",
        action="store_true",
        help="Skip Qwen model inference.",
    )
    parser.add_argument(
        "--skip-af",
        action="store_true",
        help="Skip AF-Next model inference.",
    )
    parser.add_argument(
        "--keep-models-loaded",
        action="store_true",
        help="Keep models in GPU memory throughout (faster but uses more VRAM).",
    )
    return parser.parse_args()


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_key: str) -> Tuple[Any, Any]:
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]

    model_path = MODEL_DIRS[model_key]
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型目录不存在：{model_path}。请先运行 `python deploy_models.py` 下载模型。"
        )

    print(f"[加载模型] {model_key} ...")

    if model_key == "Qwen3-Omni-Captioner":
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            dtype="auto",
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        if not torch.cuda.is_available():
            model = model.to(_device())
    elif model_key == "AF-Next-Captioner":
        from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor

        processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
        model = AudioFlamingoNextForConditionalGeneration.from_pretrained(
            str(model_path),
            trust_remote_code=True,
            torch_dtype=_dtype(),
            device_map="auto" if torch.cuda.is_available() else None,
        ).eval()
        if not torch.cuda.is_available():
            model = model.to(_device())
    else:
        raise ValueError(f"未知模型：{model_key}")

    _MODEL_CACHE[model_key] = (model, processor)
    print(f"[加载完成] {model_key} | 设备: {_device()}")
    return model, processor


def _unload_model(model_key: str):
    if model_key in _MODEL_CACHE:
        model, _ = _MODEL_CACHE.pop(model_key)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[卸载模型] {model_key}")


def _qwen_gen_output_to_new_tokens(gen_out: Any, prompt_len: int) -> torch.Tensor:
    if isinstance(gen_out, (list, tuple)):
        gen_out = gen_out[0]
    if isinstance(gen_out, str):
        raise TypeError(
            "模型 generate 返回了 str，当前环境与 Qwen3-Omni 预期不一致。"
        )
    if torch.is_tensor(gen_out):
        return gen_out[:, prompt_len:]
    seq = getattr(gen_out, "sequences", None)
    if seq is None or not torch.is_tensor(seq):
        raise TypeError(f"无法解析 Qwen generate 输出类型: {type(gen_out)}")
    return seq[:, prompt_len:]


def _build_conversation(model_key: str, audio_path: str) -> List:
    if model_key == "Qwen3-Omni-Captioner":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            }
        ]
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPTS[model_key]},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
    ]


def _infer_qwen(audio_path: str, max_new_tokens: int) -> Tuple[str, Optional[str]]:
    model, processor = _load_model("Qwen3-Omni-Captioner")
    conversation = _build_conversation("Qwen3-Omni-Captioner", audio_path)

    from qwen_omni_utils import process_mm_info

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, _, _ = process_mm_info(conversation, use_audio_in_video=False)
    batch = processor(
        text=text,
        audio=audios,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=False,
    )
    batch = batch.to(model.device).to(model.dtype)

    with torch.inference_mode():
        gen_out = model.generate(
            **batch,
            thinker_max_new_tokens=int(max_new_tokens),
            return_audio=False,
        )

    prompt_len = batch["input_ids"].shape[1]
    generated_ids = _qwen_gen_output_to_new_tokens(gen_out, prompt_len)

    text_out = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text_out or None, None


def _infer_af_next(audio_path: str, max_new_tokens: int) -> Tuple[str, Optional[str]]:
    model, processor = _load_model("AF-Next-Captioner")
    conversation = _build_conversation("AF-Next-Captioner", audio_path)

    batch = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    if torch.cuda.is_available():
        batch = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in batch.items()}
    else:
        batch = {k: v.to(_device()) if hasattr(v, "to") else v for k, v in batch.items()}

    if "input_features" in batch and hasattr(batch["input_features"], "to"):
        batch["input_features"] = batch["input_features"].to(model.dtype)

    with torch.inference_mode():
        generated = model.generate(
            **batch,
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=1.2,
        )

    prompt_len = batch["input_ids"].shape[1]
    completion = generated[:, prompt_len:]
    text_out = processor.batch_decode(
        completion,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return text_out or None, None


def load_processed_files(output_file: Path) -> set:
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if data.get("status") == "success":
                        processed.add(data.get("filename"))
                except json.JSONDecodeError:
                    continue
    return processed


def load_source_jsonl(source_jsonl: str) -> Dict[str, Dict]:
    if not source_jsonl:
        return {}
    path = Path(source_jsonl)
    if not path.exists():
        print(f"[警告] source-jsonl 不存在: {source_jsonl}")
        return {}

    manifest = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                filename = entry.get("filename")
                if filename:
                    manifest[filename] = entry
            except json.JSONDecodeError:
                continue
    return manifest


def collect_audio_files(input_root: Path, processed: set, source_manifest: Dict) -> List[Path]:
    tasks = []

    audio_extensions = (".wav", ".mp3", ".flac", ".ogg", ".m4a")

    for root, _, files in os.walk(input_root):
        for file_name in files:
            if file_name.lower().endswith(audio_extensions):
                full_path = Path(root) / file_name
                relative_path = Path(os.path.relpath(full_path, input_root)).as_posix()
                if relative_path not in processed:
                    tasks.append(full_path)

    return sorted(tasks)


def process_single_file(
    audio_path: Path,
    input_root: Path,
    args,
    source_manifest: Dict,
) -> Dict[str, Any]:
    relative_path = Path(os.path.relpath(audio_path, input_root)).as_posix()

    result = {
        "filename": relative_path,
        "status": "success",
        "qwen_dense_caption": None,
        "af_next_dense_caption": None,
        "qwen_error": None,
        "af_error": None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    source_meta = source_manifest.get(relative_path, {})
    if source_meta:
        result["source_meta_data"] = source_meta.get("meta_data")

    qwen_caption = None
    af_caption = None

    if not args.skip_qwen:
        try:
            qwen_caption, _ = _infer_qwen(str(audio_path), args.qwen_max_tokens)
            result["qwen_dense_caption"] = qwen_caption
        except Exception as exc:
            result["qwen_error"] = str(exc)
            print(f"[Qwen 错误] {relative_path}: {exc}")

    if not args.skip_af:
        try:
            af_caption, _ = _infer_af_next(str(audio_path), args.af_max_tokens)
            result["af_next_dense_caption"] = af_caption
        except Exception as exc:
            result["af_error"] = str(exc)
            print(f"[AF-Next 错误] {relative_path}: {exc}")

    if qwen_caption is None and af_caption is None:
        result["status"] = "error"
        result["error_msg"] = "两个模型均推理失败"

    return result


def run_stage2_batch(args=None):
    if args is None:
        args = parse_args()

    input_root = Path(args.input_folder)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed = load_processed_files(output_file)
    source_manifest = load_source_jsonl(args.source_jsonl)
    tasks = collect_audio_files(input_root, processed, source_manifest)

    if args.limit > 0:
        tasks = tasks[:args.limit]

    total = len(tasks)
    print(f"=== Stage 2 密集标注启动: {total} 条音频 ===")
    print(f"设备: {_device()} | Qwen tokens: {args.qwen_max_tokens} | AF tokens: {args.af_max_tokens}")
    print(f"Skip Qwen: {args.skip_qwen} | Skip AF: {args.skip_af} | Keep loaded: {args.keep_models_loaded}")

    if total == 0:
        print("[完成] 无待处理文件")
        return

    batch_size = args.batch_size
    keep_loaded = args.keep_models_loaded

    progress_bar = tqdm(tasks, desc="Dense Captioning")

    for idx, audio_path in enumerate(progress_bar):
        result = process_single_file(audio_path, input_root, args, source_manifest)

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        progress_bar.set_postfix(
            qwen="✓" if result.get("qwen_dense_caption") else "✗",
            af="✓" if result.get("af_next_dense_caption") else "✗",
        )

        if not keep_loaded and (idx + 1) % batch_size == 0:
            if not args.skip_qwen:
                _unload_model("Qwen3-Omni-Captioner")
            if not args.skip_af:
                _unload_model("AF-Next-Captioner")

    if not keep_loaded:
        if not args.skip_qwen:
            _unload_model("Qwen3-Omni-Captioner")
        if not args.skip_af:
            _unload_model("AF-Next-Captioner")

    print(f"\n[完成] Stage 2 结束，输出: {output_file}")


if __name__ == "__main__":
    run_stage2_batch()