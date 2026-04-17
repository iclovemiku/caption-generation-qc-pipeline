"""
元数据规范化工具。

提供 safe_string 与 normalize_meta_data 等函数，
用于统一处理 API 返回的 JSON 结构。
"""

import copy
from typing import Any, Dict, Optional


def safe_string(value: Any) -> str:
    """
    安全转换为字符串，避免 None 或异常类型。
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value)


def normalize_meta_data(
    meta_data: Dict[str, Any],
    source_hint: Optional[Dict[str, Any]] = None,
    backfill_from_source: bool = False,
    emit_source_emotion_hint: bool = True,
) -> Dict[str, Any]:
    """
    规范化元数据结构，统一字段命名与类型。

    Args:
        meta_data: API 返回的原始 JSON
        source_hint: 上游来源提供的弱提示字段
        backfill_from_source: 是否从 source_hint 回填缺失字段
        emit_source_emotion_hint: 是否写入 source_hint 中的情绪标签

    Returns:
        规范化后的元数据字典
    """
    normalized = copy.deepcopy(meta_data)

    required_fields = [
        "caption",
        "audio_caption",
        "transcript",
        "speech_characteristics",
        "scene_inference",
        "speaker_profile",
        "core_intent",
        "sqa_compatibility",
    ]

    for field in required_fields:
        normalized[field] = safe_string(normalized.get(field))

    if normalized.get("sqa_compatibility") not in ["High", "Medium", "Low"]:
        normalized["sqa_compatibility"] = "Low"

    if backfill_from_source and source_hint:
        backfill_fields = [
            ("transcript", "transcript"),
            ("emotion_label", "emotion_label"),
        ]
        for target_key, source_key in backfill_fields:
            if not normalized.get(target_key):
                source_value = safe_string(source_hint.get(source_key))
                if source_value:
                    normalized[target_key] = source_value

    if emit_source_emotion_hint and source_hint:
        emotion_label = safe_string(source_hint.get("emotion_label"))
        if emotion_label:
            normalized["source_emotion_hint"] = emotion_label

    return normalized