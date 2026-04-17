from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import torch
from transformers import AudioFlamingoNextForConditionalGeneration, AutoProcessor
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor


PROJECT_ROOT = Path(__file__).resolve().parent
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

# 按模型名分别缓存，便于一次会话内两个模型都保留在显存（显存不足时可改为用完即卸载）
_MODEL_CACHE: Dict[str, Tuple[Any, Any]] = {}


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

    if model_key == "Qwen3-Omni-Captioner":
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
    return model, processor


def _qwen_gen_output_to_new_tokens(gen_out: Any, prompt_len: int) -> torch.Tensor:
    if isinstance(gen_out, (list, tuple)):
        gen_out = gen_out[0]
    if isinstance(gen_out, str):
        raise TypeError(
            "模型 generate 返回了 str，当前环境与 Qwen3-Omni 预期不一致；请确认 transformers 为支持 Qwen3-Omni 的版本。"
        )
    if torch.is_tensor(gen_out):
        return gen_out[:, prompt_len:]
    seq = getattr(gen_out, "sequences", None)
    if seq is None or not torch.is_tensor(seq):
        raise TypeError(f"无法解析 Qwen generate 输出类型: {type(gen_out)}（期望 Tensor 或带 tensor .sequences 的对象）")
    return seq[:, prompt_len:]


def _build_conversation(model_key: str, audio_path: str, custom_prompt: Optional[str]) -> list:
    if model_key == "Qwen3-Omni-Captioner":
        return [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                ],
            }
        ]

    user_prompt = (custom_prompt or "").strip() or PROMPTS[model_key]
    return [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
    ]


def _infer_one(model_key: str, audio_file: str, custom_prompt: str, max_new_tokens: int) -> Tuple[str, str]:
    model, processor = _load_model(model_key)
    conversation = _build_conversation(model_key, audio_file, custom_prompt)

    if model_key == "Qwen3-Omni-Captioner":
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
    else:
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

    info = f"设备：{_device()}\n权重：{MODEL_DIRS[model_key]}"
    return text_out or "(模型返回空文本)", info


def run_both_models(
    audio_file: Optional[str],
    af_prompt: str,
    qwen_max: int,
    af_max: int,
) -> Tuple[str, str, str]:
    """依次运行两个模型，并排展示结果。"""
    if not audio_file:
        return "请先上传音频。", "请先上传音频。", ""

    qwen_text, qwen_info = "", ""
    af_text, af_info = "", ""

    try:
        qwen_text, qwen_info = _infer_one("Qwen3-Omni-Captioner", audio_file, "", int(qwen_max))
    except Exception as exc:  # noqa: BLE001
        qwen_text = f"Qwen 推理失败：{exc}"

    try:
        af_text, af_info = _infer_one("AF-Next-Captioner", audio_file, af_prompt, int(af_max))
    except Exception as exc:  # noqa: BLE001
        af_text = f"AF-Next 推理失败：{exc}"

    meta = (
        f"[Qwen3-Omni-Captioner]\n{qwen_info}\n\n"
        f"[AF-Next-Captioner]\n{af_info}"
    )
    return qwen_text, af_text, meta


def run_single_model(
    model_key: str,
    audio_file: Optional[str],
    custom_prompt: str,
    max_new_tokens: int,
) -> Tuple[str, str]:
    """可选：只跑一个模型（调试）。"""
    if not audio_file:
        return "请先上传音频文件。", ""
    try:
        # Qwen 不传文本 prompt；AF-Next 用 custom_prompt
        prompt = "" if model_key == "Qwen3-Omni-Captioner" else custom_prompt
        text_out, info = _infer_one(model_key, audio_file, prompt, int(max_new_tokens))
        return text_out, info
    except Exception as exc:  # noqa: BLE001
        return f"推理失败：{exc}", ""


with gr.Blocks(title="Caption Pipeline UI") as demo:
    gr.Markdown("# Audio Caption 双模型对比")
    gr.Markdown(
        "上传一段音频后点击 **同时生成**，将依次运行 **Qwen3-Omni-Captioner** 与 **AF-Next-Captioner**，"
        "并在左右两栏同时查看结果。\n\n"
        "*说明：Qwen Captioner 仅使用音频，不使用下方 Prompt；Prompt 仅作用于 AF-Next。*"
    )

    audio_file = gr.Audio(type="filepath", label="上传音频")

    af_prompt = gr.Textbox(
        label="AF-Next 可选 Prompt（不填则使用默认长描述提示）",
        placeholder="例如：请输出带时间线的详细描述，并完整转写人声。",
        lines=3,
    )

    with gr.Row():
        qwen_max = gr.Slider(
            minimum=128,
            maximum=4096,
            value=1024,
            step=64,
            label="Qwen max_new_tokens",
        )
        af_max = gr.Slider(
            minimum=128,
            maximum=4096,
            value=1024,
            step=64,
            label="AF-Next max_new_tokens",
        )

    run_both_btn = gr.Button("同时生成（两个模型）", variant="primary")

    with gr.Row():
        qwen_out = gr.Textbox(label="Qwen3-Omni-Captioner 输出", lines=18)
        af_out = gr.Textbox(label="AF-Next-Captioner 输出", lines=18)

    run_meta = gr.Textbox(label="运行信息（路径与设备）", lines=6)

    run_both_btn.click(
        fn=run_both_models,
        inputs=[audio_file, af_prompt, qwen_max, af_max],
        outputs=[qwen_out, af_out, run_meta],
    )

    gr.Markdown("---\n### 可选：仅运行单个模型")
    with gr.Row():
        model_key = gr.Dropdown(
            choices=list(MODEL_DIRS.keys()),
            value="AF-Next-Captioner",
            label="选择模型",
        )
        max_single = gr.Slider(
            minimum=128,
            maximum=4096,
            value=1024,
            step=64,
            label="max_new_tokens",
        )
    single_prompt = gr.Textbox(label="Prompt（仅 AF-Next 生效）", lines=2)
    run_single_btn = gr.Button("仅生成所选模型")
    single_out = gr.Textbox(label="单模型输出", lines=12)
    single_info = gr.Textbox(label="单模型运行信息", lines=3)

    run_single_btn.click(
        fn=run_single_model,
        inputs=[model_key, audio_file, single_prompt, max_single],
        outputs=[single_out, single_info],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
