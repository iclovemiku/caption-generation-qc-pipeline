import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import urllib3


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.metadata_contract import normalize_meta_data, safe_string  # noqa: E402


DEFAULT_INPUT_FOLDER = str(PROJECT_ROOT / "data" / "01_raw_audio")
DEFAULT_OUTPUT_FILE = str(PROJECT_ROOT / "data" / "02_base_captions" / "base_captions.jsonl")
DEFAULT_API_URL = "https://az.gptplus5.com/v1/chat/completions"
DEFAULT_API_KEY = "sk-6GrqRfDmmk5NlJPRwujFjOdiAlULir5CYEtL6HVBMgeCRjtg"
DEFAULT_MODEL_NAME = "gemini-3-flash-preview"
DEFAULT_PROXY = "http://x00957272:%40Xh200225@proxyau.huawei.com:8080/"

MIME_BY_SUFFIX = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".m4a": "audio/mp4",
}

# LEGACY_SYSTEM_PROMPT = """你是一位专业的音频与情景分析专家 (Audio Scene Analyst)。
# 请分析音频并返回严格的 JSON 对象（不要使用 Markdown，不要包含任何其他解释性文字）：

# 1. "caption": (String) 简体中文。融合描述说话人、内容、环境音。
# 2. "audio_caption": (String) 简体中文。仅描述背景音和环境。
# 3. "transcript": (String) 听写逐字稿。如果是噪音或无人声，输出"无"。
# 4. "speech_characteristics": (String) 简体中文。描述说话人特征。禁止字典格式。
# 5. "scene_inference": (String) 简体中文。推断物理场景。
# """

# EMOTION_AWARE_SYSTEM_PROMPT = """你是一位专业的中文音频与情绪分析专家。
# 请分析音频并只返回严格的 JSON 对象，不要使用 Markdown，不要输出额外解释。

# 输出字段必须包含：
# 1. "caption": 简体中文。描述说话人正在表达什么、情绪落点是什么、以及和环境声之间的关系。
# 2. "audio_caption": 简体中文。只描述背景音、环境声、空间感、混响、底噪、人群、交通、室内外线索等，不要复述 transcript。
# 3. "transcript": 保持原始说话语言。若音频是中文，请返回中文逐字稿；若无人声或无法可靠听清，输出"无"。
# 4. "speech_characteristics": 简体中文。必须描述至少两类可听见的人声特征，例如情绪强度、语速、停顿、呼吸、笑意、哭腔、犹豫、压低嗓音、爆发感、互动张力。禁止字典格式。
# 5. "scene_inference": 简体中文。基于音频推测物理场景；即使信息不足也不能留空，需要给出带不确定性的场景判断。

# 额外要求：
# - 优先写“可听见”的事实，不要只根据字面内容空泛概括情绪。
# - 如果给了上游 transcript / emotion / background 提示，只能作为弱参考，必须以音频本身为准。
# - 对情绪保持克制判断，多使用“可能、像是、听起来似乎、带有一点”等表达，避免绝对化。
# """
EMOTION_AWARE_SYSTEM_PROMPT = """你是一位专业的中文音频与情绪心理分析专家。
请分析音频并只返回严格的 JSON 对象，不要使用 Markdown，不要输出额外解释。

输出字段必须包含：
1. "caption": 简体中文。融合描述说话人、内容、环境音。
2. "audio_caption": 简体中文。只描述背景音、环境声、空间感、混响、底噪等，不要复述 transcript。
3. "transcript": 保持原始说话语言。若音频是中文，请返回中文逐字稿；若无人声或无法可靠听清，输出"无"。
4. "speech_characteristics": 简体中文。必须描述至少两类可听见的人声特征（如情绪强度、语速、停顿、呼吸、哭腔、叹气、发抖、咬字力度等）。禁止字典格式。
5. "scene_inference": 简体中文。基于音频推测物理场景。

【👇为下游 SQA 共情任务新增的字段👇】
6. "speaker_profile": 简体中文。推测说话人的基本画像（如：年轻女性、中年男性、儿童），若多人则描述主发言人。
7. "core_intent": 简体中文。推测说话人的底层心理诉求。**要求语言精炼、一针见血，像人类的口语化概括，拒绝冗长拗口的书面学术腔调。**（例如：需要情绪宣泄、寻求认同、分享极度喜悦、客观陈述事实、绝望求助）。
8. "sqa_compatibility": (String) 只能在 ["High", "Medium", "Low"] 中选一。
   - 评估这段音频是否适合伪装成“用户发给AI的单方面语音留言”。
   - High: 独白、倾诉、个人情绪发泄（非常适合伪装成用户语音）。
   - Medium: 带有部分互动，但可以勉强视为对AI说话。
   - Low: 多人对谈、新闻播报、影视剧对白、背景嘈杂的无关发言（完全不适合对AI说）。

额外要求：
- **致命红线：** `caption` 和 `core_intent` 的描述【绝对不能】与 `transcript` 的字面语义产生逻辑矛盾（例如：transcript 是拒绝，caption 绝不能写成渴望）。
- 对情绪保持克制判断，多使用“可能、像是、听起来似乎”等表达。
- 必须以音频本身为准。
"""


jsonl_lock = threading.Lock()
urllib3.disable_warnings()

from local_api_logger import wrap_requests_call


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate AF-chat metadata captions from audio clips, with optional emotional-audio source hints."
    )
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER, help="Folder containing audio files.")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="Metadata JSONL output path.")
    parser.add_argument("--source-manifest", default="", help="Optional clip-level JSONL manifest used as weak hints.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Chat completion API endpoint.")
    parser.add_argument("--api-key", default=os.environ.get("AFCHAT_API_KEY", DEFAULT_API_KEY), help="API key.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Model name.")
    parser.add_argument("--max-workers", type=int, default=20, help="Worker count.")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries per file.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="Per-request timeout in seconds.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick pilot runs.")
    parser.add_argument(
        "--prompt-mode",
        choices=("legacy", "emotion-aware"),
        default="emotion-aware",
        help="Prompt variant used for caption generation.",
    )
    parser.add_argument(
        "--backfill-from-manifest",
        action="store_true",
        help="Backfill transcript/emotion hints from the source manifest when the model leaves them weak or missing.",
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("AFCHAT_HTTP_PROXY", DEFAULT_PROXY),
        help="Optional HTTP(S) proxy. Empty string disables proxy injection.",
    )
    return parser.parse_args()


def apply_proxy(proxy: str):
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy


def encode_audio(file_path: Path):
    with open(file_path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def extract_json_content(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    raise ValueError("未找到有效的 JSON 对象")


def load_processed_files(output_file: Path):
    processed = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("status") == "success":
                    processed.add(data.get("filename"))
    return processed


def load_source_manifest(manifest_path: str):
    if not manifest_path:
        return {}

    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Source manifest not found: {manifest_path}")

    manifest_map = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            keys = [
                safe_string(entry.get("relative_path")),
                safe_string(entry.get("filename")),
                Path(safe_string(entry.get("audio_path"))).name if safe_string(entry.get("audio_path")) else "",
            ]
            for key in keys:
                if key and key not in manifest_map:
                    manifest_map[key] = entry
    return manifest_map


def get_source_hint(relative_path: str, full_file_path: Path, manifest_map):
    lookup_keys = [relative_path, Path(relative_path).as_posix(), full_file_path.name]
    for key in lookup_keys:
        if key in manifest_map:
            return manifest_map[key]
    return {}


def get_system_prompt(prompt_mode: str):
    return EMOTION_AWARE_SYSTEM_PROMPT if prompt_mode == "emotion-aware" else LEGACY_SYSTEM_PROMPT


def build_user_text(source_hint):
    prompt_lines = ["请分析这段音频，并严格返回 JSON 对象。"]
    if source_hint:
        hint_fields = []
        hint_mapping = [
            ("source_name", "来源"),
            ("transcript", "上游 transcript"),
            ("emotion_label", "上游情绪标签"),
            ("affect_profile", "上游情绪轮廓"),
            ("background_hint", "上游背景提示"),
            ("language", "语言提示"),
        ]
        for field, label in hint_mapping:
            value = safe_string(source_hint.get(field))
            if value:
                hint_fields.append(f"- {label}: {value}")
        if hint_fields:
            prompt_lines.append("以下是上游来源提供的弱提示，只能作为参考，必须以音频本身为准：")
            prompt_lines.extend(hint_fields)
    return "\n".join(prompt_lines)


def build_payload(file_path: Path, args, source_hint):
    base64_audio = encode_audio(file_path)
    mime_type = MIME_BY_SUFFIX.get(file_path.suffix.lower(), "audio/wav")
    audio_data_uri = f"data:{mime_type};base64,{base64_audio}"

    return {
        "model": args.model,
        "messages": [
            {"role": "system", "content": get_system_prompt(args.prompt_mode)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_text(source_hint)},
                    {"type": "image_url", "image_url": {"url": audio_data_uri}},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": 2048,
    }


def parse_response_json(response):
    if isinstance(response, dict):
        return response
    if hasattr(response, "json"):
        return response.json()
    return json.loads(response.text)


def process_single_file(full_file_path: Path, input_root: Path, output_file: Path, args, manifest_map):
    relative_path = os.path.relpath(full_file_path, input_root)
    source_hint = get_source_hint(relative_path, full_file_path, manifest_map)

    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {args.api_key}"}
        payload = build_payload(full_file_path, args=args, source_hint=source_hint)
    except Exception as exc:
        return f"❌ 文件读取错误: {exc}"

    last_error = None
    for attempt in range(args.max_retries):
        try:
            response = wrap_requests_call(
                model=args.model,
                url=args.api_url,
                headers=headers,
                payload=payload,
                user="afchat_caption_generation",
                verify=False,
                timeout=args.timeout_sec,
            )
            response_json = parse_response_json(response)
            if "error" in response_json:
                raise RuntimeError(f"API Error: {response_json['error']}")

            raw_content = response_json["choices"][0]["message"]["content"]
            meta_data = extract_json_content(raw_content)
            normalized_meta = normalize_meta_data(
                meta_data,
                source_hint=source_hint,
                backfill_from_source=args.backfill_from_manifest,
                # 上游 emotion_label 仍通过 build_user_text 给模型作弱参考，但落盘不写 emotion_hint
                emit_source_emotion_hint=False,
            )

            final_record = {
                "filename": Path(relative_path).as_posix(),
                "status": "success",
                "meta_data": normalized_meta,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with jsonl_lock:
                with open(output_file, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(final_record, ensure_ascii=False) + "\n")
            return f"✅ {normalized_meta.get('scene_inference', 'OK')[:18]}..."
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"⚠️ [{relative_path}] 第 {attempt + 1} 次失败 ({exc}) -> {wait_time}秒后重试")
                time.sleep(wait_time)

    error_msg = str(last_error) if last_error else "unknown_error"
    if "Read timed out" in error_msg:
        error_msg = f"最终超时 ({args.timeout_sec}s x {args.max_retries})"

    error_record = {
        "filename": Path(relative_path).as_posix(),
        "status": "error",
        "error_msg": error_msg,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with jsonl_lock:
        with open(output_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(error_record, ensure_ascii=False) + "\n")
    return f"❌ 彻底失败: {error_msg}"


def collect_tasks(input_root: Path, processed):
    tasks = []
    for root, _, files in os.walk(input_root):
        for file_name in files:
            if file_name.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                full_path = Path(root) / file_name
                relative_path = Path(os.path.relpath(full_path, input_root)).as_posix()
                if relative_path not in processed:
                    tasks.append(full_path)
    return sorted(tasks)


def main():
    args = parse_args()
    apply_proxy(args.proxy)

    input_root = Path(args.input_folder)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    processed = load_processed_files(output_file)
    manifest_map = load_source_manifest(args.source_manifest)
    tasks = collect_tasks(input_root, processed)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"=== Caption 任务启动: {len(tasks)} 条 | 模型: {args.model} ===")
    print(
        f"配置: 线程={args.max_workers} | 超时={args.timeout_sec}s | 重试={args.max_retries}次 | "
        f"Prompt={args.prompt_mode}"
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_single_file, task, input_root, output_file, args, manifest_map): task
            for task in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            print(f"[{index}/{len(tasks)}] {futures[future].name} -> {future.result()}")


if __name__ == "__main__":
    main()