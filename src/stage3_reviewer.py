"""
Stage 3: 一致性审查与环境过滤（本地文本大模型裁判）。

职责：
- 交叉比对 Base JSON (Stage 1) 与双 Dense Caption (Stage 2)
- 幻觉校验
- 环境噪声过滤
- 质量评分

用法:
    python -m src.stage3_reviewer
    python -m src.stage3_reviewer --judge-model qwen32b
    python -m src.stage3_reviewer --judge-model llama70b
    python -m src.stage3_reviewer --judge-model qwen35b-opus
"""

import argparse
import gc
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

JUDGE_MODEL_ROOT = PROJECT_ROOT / "models" / "judge"

JUDGE_MODELS = {
    "qwen32b": {
        "path": JUDGE_MODEL_ROOT / "deepseek-r1-qwen-32b",
        "name": "DeepSeek-R1-Distill-Qwen-32B",
    },
    "llama70b": {
        "path": JUDGE_MODEL_ROOT / "deepseek-r1-llama-70b",
        "name": "DeepSeek-R1-Distill-Llama-70B",
    },
    "qwen35b-opus": {
        "path": JUDGE_MODEL_ROOT / "qwen35-opus-35b",
        "name": "Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled",
    },
}

DEFAULT_BASE_CAPTIONS = str(PROJECT_ROOT / "data" / "02_base_captions" / "base_captions.jsonl")
DEFAULT_DENSE_CAPTIONS = str(PROJECT_ROOT / "data" / "03_dense_captions" / "dense_captions.jsonl")
DEFAULT_OUTPUT_FILE = str(PROJECT_ROOT / "data" / "04_cleaned_metadata" / "cleaned_metadata.jsonl")

_JUDGE_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}
_jsonl_lock = threading.Lock()


REVIEW_SYSTEM_PROMPT = """你是一位严格的音频数据质量审查专家。
你的任务是审查音频描述数据的一致性、准确性和质量。

你需要对比以下来源的描述：
1. **Base Caption**：来自远程多模态 API (Gemini等) 的基础描述
2. **Qwen Dense Caption**：来自本地 Qwen3-Omni-Captioner 的密集描述
3. **AF-Next Dense Caption**：来自本地 AF-Next-Captioner 的密集描述

请分析并返回严格的 JSON 对象（不要使用 Markdown），包含以下字段：

{
  "consistency_score": (整数 1-10) 多个来源描述的一致性评分
  "hallucination_detected": (布尔) 是否检测到明显幻觉（描述了不存在的内容）
  "hallucination_details": (字符串) 如果检测到幻觉，描述具体问题
  "quality_score": (整数 1-10) 整体数据质量评分
  "keep_record": (布尔) 是否建议保留这条记录用于下游训练
  "rejection_reason": (字符串) 如果不建议保留，说明原因
  "best_caption_source": (字符串 "base"|"qwen"|"af_next"|"merged") 哪个来源的描述最准确
  "review_notes": (字符串) 审查备注和建议
}

评分标准：
- consistency_score: 10=完全一致，7=基本一致但有细节差异，4=有明显矛盾，1=完全冲突
- quality_score: 10=高质量可用，7=可用但有小问题，4=质量较低，1=不可用
- keep_record: 只有 quality_score >= 5 且无明显幻觉时才为 true

重要规则：
- transcript 字段的听写内容如果差异大，需要特别关注
- scene_inference 如果与音频内容明显不符，可能是幻觉
- 如果所有来源都描述了某个细节，则认为该细节可信
- 如果只有单一来源提到某个细节且其他来源没提到，可能是幻觉
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 3: 使用本地裁判模型审查音频描述数据。"
    )
    parser.add_argument(
        "--base-captions",
        default=DEFAULT_BASE_CAPTIONS,
        help="Stage 1 输出的 base_captions.jsonl",
    )
    parser.add_argument(
        "--dense-captions",
        default=DEFAULT_DENSE_CAPTIONS,
        help="Stage 2 输出的 dense_captions.jsonl",
    )
    parser.add_argument(
        "--output-file",
        default=DEFAULT_OUTPUT_FILE,
        help="输出 cleaned_metadata.jsonl",
    )
    parser.add_argument(
        "--judge-model",
        choices=list(JUDGE_MODELS.keys()),
        default="qwen32b",
        help="选择裁判模型",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="并行工作数（大模型推理建议设为1）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="裁判模型最大生成长度",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="测试时限制处理数量",
    )
    parser.add_argument(
        "--batch-clear",
        type=int,
        default=10,
        help="每处理 N 条后清理 GPU 缓存",
    )
    parser.add_argument(
        "--force-think",
        action="store_true",
        help="强制模型以思考块开头（DeepSeek R1 推荐开启）",
    )
    return parser.parse_args()


def _dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_judge_model(model_key: str) -> Tuple[Any, Any]:
    if model_key in _JUDGE_MODEL_CACHE:
        return _JUDGE_MODEL_CACHE[model_key]

    model_info = JUDGE_MODELS[model_key]
    model_path = model_info["path"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"裁判模型目录不存在：{model_path}\n请先运行: python deploy_judge_models.py --model {model_key}"
        )

    print(f"[加载裁判模型] {model_info['name']} ...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        trust_remote_code=True,
        torch_dtype=_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    if not torch.cuda.is_available():
        model = model.to(_device())

    _JUDGE_MODEL_CACHE[model_key] = (model, tokenizer)
    print(f"[加载完成] {model_info['name']} | 设备: {_device()}")
    return model, tokenizer


def unload_judge_model(model_key: str):
    if model_key in _JUDGE_MODEL_CACHE:
        model, tokenizer = _JUDGE_MODEL_CACHE.pop(model_key)
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[卸载裁判模型] {model_key}")


def load_jsonl(file_path: str) -> Dict[str, Dict]:
    data_map = {}
    path = Path(file_path)
    if not path.exists():
        print(f"[警告] 文件不存在: {file_path}")
        return data_map

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                filename = entry.get("filename")
                if filename:
                    data_map[filename] = entry
            except json.JSONDecodeError:
                continue

    print(f"[加载] {file_path}: {len(data_map)} 条记录")
    return data_map


def build_review_prompt(
    base_data: Optional[Dict],
    dense_data: Optional[Dict],
    force_think: bool = False,
) -> str:
    prompt_parts = [REVIEW_SYSTEM_PROMPT, "\n\n以下是待审查的数据：\n"]

    if base_data and base_data.get("status") == "success":
        meta = base_data.get("meta_data", {})
        prompt_parts.append("【Base Caption 来源】\n")
        prompt_parts.append(f"- filename: {base_data.get('filename')}\n")
        prompt_parts.append(f"- caption: {meta.get('caption', '无')}\n")
        prompt_parts.append(f"- audio_caption: {meta.get('audio_caption', '无')}\n")
        prompt_parts.append(f"- transcript: {meta.get('transcript', '无')}\n")
        prompt_parts.append(f"- speech_characteristics: {meta.get('speech_characteristics', '无')}\n")
        prompt_parts.append(f"- scene_inference: {meta.get('scene_inference', '无')}\n")
        prompt_parts.append(f"- speaker_profile: {meta.get('speaker_profile', '无')}\n")
        prompt_parts.append(f"- core_intent: {meta.get('core_intent', '无')}\n")
        prompt_parts.append(f"- sqa_compatibility: {meta.get('sqa_compatibility', '无')}\n")
        prompt_parts.append("\n")
    else:
        prompt_parts.append("【Base Caption 来源】数据缺失或失败\n\n")

    if dense_data and dense_data.get("status") == "success":
        prompt_parts.append("【Qwen Dense Caption 来源】\n")
        qwen_cap = dense_data.get("qwen_dense_caption")
        prompt_parts.append(f"- dense_caption: {qwen_cap if qwen_cap else '无'}\n")
        prompt_parts.append("\n")

        prompt_parts.append("【AF-Next Dense Caption 来源】\n")
        af_cap = dense_data.get("af_next_dense_caption")
        prompt_parts.append(f"- dense_caption: {af_cap if af_cap else '无'}\n")
        prompt_parts.append("\n")
    else:
        prompt_parts.append("【Dense Caption 来源】数据缺失或失败\n\n")

    prompt_parts.append("请审查以上数据并返回 JSON 格式的审查结果。")

    user_prompt = "".join(prompt_parts)

    if force_think:
        user_prompt = "请仔细分析以下数据。\n\n" + user_prompt

    return user_prompt


def extract_json_from_response(text: str) -> Dict:
    text = text.strip()
    if text.startswith(""):
        think_end = text.find("")
        if think_end != -1:
            text = text[think_end + 5:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    import re
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    return {}


def run_judge_inference(
    model_key: str,
    prompt: str,
    max_new_tokens: int,
    force_think: bool = False,
) -> Dict:
    model, tokenizer = load_judge_model(model_key)

    messages = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    if force_think and "deepseek" in model_key:
        if not text.endswith("\n"):
            text += "\n"

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.05,
        )

    generated_text = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    return extract_json_from_response(generated_text)


def process_single_record(
    filename: str,
    base_data: Optional[Dict],
    dense_data: Optional[Dict],
    args,
) -> Dict:
    result = {
        "filename": filename,
        "status": "success",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_model": JUDGE_MODELS[args.judge_model]["name"],
    }

    if base_data:
        result["source_base_meta"] = base_data.get("meta_data")
    if dense_data:
        result["source_qwen_caption"] = dense_data.get("qwen_dense_caption")
        result["source_af_caption"] = dense_data.get("af_next_dense_caption")

    prompt = build_review_prompt(base_data, dense_data, args.force_think)

    try:
        review_result = run_judge_inference(
            args.judge_model,
            prompt,
            args.max_new_tokens,
            args.force_think,
        )

        if not review_result:
            result["status"] = "error"
            result["error_msg"] = "裁判模型未能返回有效 JSON"
            return result

        result["review_result"] = review_result

        quality_score = review_result.get("quality_score", 0)
        hallucination = review_result.get("hallucination_detected", True)
        keep_record = review_result.get("keep_record", False)

        if quality_score >= 5 and not hallucination and keep_record:
            result["final_status"] = "approved"
            if base_data and base_data.get("meta_data"):
                merged_meta = base_data["meta_data"].copy()
                if dense_data:
                    merged_meta["qwen_dense_caption"] = dense_data.get("qwen_dense_caption")
                    merged_meta["af_next_dense_caption"] = dense_data.get("af_next_dense_caption")
                merged_meta["review_quality_score"] = quality_score
                merged_meta["review_consistency_score"] = review_result.get("consistency_score", 0)
                merged_meta["best_caption_source"] = review_result.get("best_caption_source", "base")
                result["final_meta_data"] = merged_meta
        else:
            result["final_status"] = "rejected"
            result["rejection_reason"] = review_result.get("rejection_reason", "质量不达标")

    except Exception as exc:
        result["status"] = "error"
        result["error_msg"] = str(exc)

    return result


def collect_filenames(base_map: Dict, dense_map: Dict) -> List[str]:
    filenames = set()
    filenames.update(base_map.keys())
    filenames.update(dense_map.keys())
    return sorted(filenames)


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


def run_stage3_batch(args=None):
    if args is None:
        args = parse_args()

    base_map = load_jsonl(args.base_captions)
    dense_map = load_jsonl(args.dense_captions)

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed = load_processed_files(output_file)
    all_filenames = collect_filenames(base_map, dense_map)

    pending = [f for f in all_filenames if f not in processed]

    if args.limit > 0:
        pending = pending[:args.limit]

    total = len(pending)

    print("=" * 60)
    print(f"Stage 3 审查启动")
    print(f"裁判模型: {JUDGE_MODELS[args.judge_model]['name']}")
    print(f"待审查: {total} 条 | 已完成: {len(processed)} 条")
    print(f"设备: {_device()} | 强制思考: {args.force_think}")
    print("=" * 60)

    if total == 0:
        print("[完成] 无待处理记录")
        return

    progress_bar = tqdm(pending, desc="审查中")

    approved_count = 0
    rejected_count = 0
    error_count = 0

    for idx, filename in enumerate(progress_bar):
        base_data = base_map.get(filename)
        dense_data = dense_map.get(filename)

        result = process_single_record(filename, base_data, dense_data, args)

        with _jsonl_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if result.get("final_status") == "approved":
            approved_count += 1
        elif result.get("final_status") == "rejected":
            rejected_count += 1
        else:
            error_count += 1

        progress_bar.set_postfix(
            approved=approved_count,
            rejected=rejected_count,
            error=error_count,
        )

        if (idx + 1) % args.batch_clear == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    unload_judge_model(args.judge_model)

    print("\n" + "=" * 60)
    print(f"[完成] Stage 3 审查结束")
    print(f"通过: {approved_count} | 拒绝: {rejected_count} | 错误: {error_count}")
    print(f"输出: {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    run_stage3_batch()