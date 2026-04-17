"""
Pipeline 可视化界面：展示音频数据处理流程的每个阶段。

用法:
    python pipeline_visualizer.py

访问:
    http://localhost:7861
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIRS = {
    "01_raw_audio": PROJECT_ROOT / "data" / "01_raw_audio",
    "02_base_captions": PROJECT_ROOT / "data" / "02_base_captions",
    "03_dense_captions": PROJECT_ROOT / "data" / "03_dense_captions",
    "04_cleaned_metadata": PROJECT_ROOT / "data" / "04_cleaned_metadata",
    "05_final_datasets": PROJECT_ROOT / "data" / "05_final_datasets",
}

PIPELINE_STAGES = {
    "Stage 1": {
        "name": "基础标注",
        "description": "调用远程多模态 API (Gemini 等) 生成基础字幕",
        "module": "src.stage1_captioner.py",
        "input": "data/01_raw_audio/*.wav",
        "output": "data/02_base_captions/base_captions.jsonl",
        "model": "Gemini / OpenAI 兼容 API",
        "fields": [
            "caption", "audio_caption", "transcript",
            "speech_characteristics", "scene_inference",
            "speaker_profile", "core_intent", "sqa_compatibility"
        ],
        "color": "#4CAF50",
    },
    "Stage 2": {
        "name": "密集标注",
        "description": "本地双模型 (Qwen3-Omni + AF-Next) 进行密集音频描述",
        "module": "src.stage2_dense_infer.py",
        "input": "data/01_raw_audio/*.wav",
        "output": "data/03_dense_captions/dense_captions.jsonl",
        "models": [
            "Qwen3-Omni-30B-A3B-Captioner",
            "Audio-Flamingo-Next-Captioner"
        ],
        "fields": ["qwen_dense_caption", "af_next_dense_caption"],
        "color": "#2196F3",
    },
    "Stage 3": {
        "name": "质量审查",
        "description": "本地裁判模型交叉验证、幻觉检测、质量评分",
        "module": "src.stage3_reviewer.py",
        "input": "Stage 1 + Stage 2 的 JSONL",
        "output": "data/04_cleaned_metadata/cleaned_metadata.jsonl",
        "models": [
            "DeepSeek-R1-Distill-Qwen-32B",
            "DeepSeek-R1-Distill-Llama-70B",
            "Qwen3.5-35B-Claude-Opus-Distilled"
        ],
        "fields": [
            "consistency_score", "hallucination_detected",
            "quality_score", "keep_record", "final_meta_data"
        ],
        "color": "#FF9800",
    },
    "Stage 4": {
        "name": "数据生成",
        "description": "生成下游训练数据：SQA-CoT / AQA-CoT / 多轮对话",
        "module": "src.stage4_generator.py",
        "input": "data/04_cleaned_metadata/*.jsonl",
        "output": [
            "data/05_final_datasets/sqa_cot/",
            "data/05_final_datasets/aqa_cot/",
            "data/05_final_datasets/multi_turn/"
        ],
        "tasks": ["SQA-CoT 单轮共情", "AQA-CoT 音频问答", "多轮对话"],
        "color": "#9C27B0",
    },
}


def count_files_in_dir(dir_path: Path, ext: str = None) -> int:
    if not dir_path.exists():
        return 0
    count = 0
    for root, _, files in os.walk(dir_path):
        for f in files:
            if ext is None or f.lower().endswith(ext):
                count += 1
    return count


def count_jsonl_records(file_path: Path) -> tuple:
    if not file_path.exists():
        return 0, 0
    count = 0
    success_count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            count += 1
            try:
                data = json.loads(line)
                if data.get("status") == "success":
                    success_count += 1
            except:
                pass
    return count, success_count


def get_pipeline_stats() -> Dict:
    stats = {}

    audio_count = count_files_in_dir(DATA_DIRS["01_raw_audio"], ".wav")
    stats["raw_audio"] = audio_count

    base_jsonl = DATA_DIRS["02_base_captions"] / "base_captions.jsonl"
    total, success = count_jsonl_records(base_jsonl)
    stats["stage1_total"] = total
    stats["stage1_success"] = success

    dense_jsonl = DATA_DIRS["03_dense_captions"] / "dense_captions.jsonl"
    total, success = count_jsonl_records(dense_jsonl)
    stats["stage2_total"] = total
    stats["stage2_success"] = success

    cleaned_jsonl = DATA_DIRS["04_cleaned_metadata"] / "cleaned_metadata.jsonl"
    total, success = count_jsonl_records(cleaned_jsonl)
    stats["stage3_total"] = total
    stats["stage3_success"] = success

    final_dir = DATA_DIRS["05_final_datasets"]
    stats["final_files"] = count_files_in_dir(final_dir, ".jsonl")

    return stats


def load_sample_record(file_path: Path, index: int = 0) -> Optional[Dict]:
    if not file_path.exists():
        return None
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except:
                pass
    if index < len(records):
        return records[index]
    return None


def format_record_display(record: Dict) -> str:
    if not record:
        return "无数据"
    return json.dumps(record, indent=2, ensure_ascii=False)


def get_mermaid_flowchart() -> str:
    return """
```mermaid
flowchart TD
    subgraph INPUT["📥 输入数据"]
        A1["01_raw_audio<br/>原始音频文件"]
    end

    subgraph S1["Stage 1: 基础标注 🌐"]
        B1["远程多模态 API<br/>Gemini / GPT"]
        B2["输出: base_captions.jsonl"]
    end

    subgraph S2["Stage 2: 密集标注 🤖"]
        C1["Qwen3-Omni-Captioner<br/>本地推理"]
        C2["AF-Next-Captioner<br/>本地推理"]
        C3["输出: dense_captions.jsonl"]
    end

    subgraph S3["Stage 3: 质量审查 🧐"]
        D1["裁判模型审查<br/>DeepSeek R1 Distill"]
        D2["交叉验证 & 幻觉检测"]
        D3["输出: cleaned_metadata.jsonl"]
    end

    subgraph S4["Stage 4: 数据生成 📦"]
        E1["SQA-CoT<br/>单轮共情数据"]
        E2["AQA-CoT<br/>音频问答数据"]
        E3["Multi-Turn<br/>多轮对话数据"]
    end

    subgraph OUTPUT["📤 最终输出"]
        F1["05_final_datasets/<br/>训练数据"]
    end

    A1 --> B1
    B1 --> B2
    A1 --> C1
    A1 --> C2
    C1 --> C3
    C2 --> C3
    B2 --> D1
    C3 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> E1
    D3 --> E2
    D3 --> E3
    E1 --> F1
    E2 --> F1
    E3 --> F1

    style INPUT fill:#e8f5e9
    style S1 fill:#c8e6c9
    style S2 fill:#bbdefb
    style S3 fill:#fff3e0
    style S4 fill:#f3e5f5
    style OUTPUT fill:#fce4ec
```
"""


def build_stage_detail_html(stage_key: str) -> str:
    stage = PIPELINE_STAGES[stage_key]
    html = f"""
    <div style="background: {stage['color']}20; padding: 20px; border-radius: 10px; border-left: 5px solid {stage['color']};">
        <h2 style="color: {stage['color']};">{stage_key}: {stage['name']}</h2>
        <p><strong>描述:</strong> {stage['description']}</p>
        <p><strong>模块:</strong> <code>{stage['module']}</code></p>
        <p><strong>输入:</strong> {stage['input']}</p>
        <p><strong>输出:</strong> {stage['output']}</p>
        <h3>使用的模型:</h3>
        <ul>
    """
    models = stage.get("models", [stage.get("model", "未指定")])
    for model in models:
        html += f"<li>{model}</li>"
    html += "</ul>"
    html += "<h3>输出字段:</h3><ul>"
    for field in stage.get("fields", []):
        html += f"<li><code>{field}</code></li>"
    html += "</ul></div>"
    return html


def create_interface():
    with gr.Blocks(title="AF-Chat Pipeline 可视化") as demo:
        gr.Markdown("""
        # AF-Chat Emotion & CoT Audio Data Pipeline
        ## 端到端多模态音频数据生产流水线
        """)

        with gr.Tab("📊 流程总览"):
            gr.Markdown(get_mermaid_flowchart())

            stats = get_pipeline_stats()
            gr.Markdown(f"""
            ### 当前数据统计

            | 阶段 | 数据量 | 状态 |
            |------|--------|------|
            | 原始音频 | {stats['raw_audio']} 个 | 📥 输入 |
            | Stage 1 基础标注 | {stats['stage1_success']}/{stats['stage1_total']} 条 | {'✅ 已完成' if stats['stage1_success'] > 0 else '⏳ 待处理'} |
            | Stage 2 密集标注 | {stats['stage2_success']}/{stats['stage2_total']} 条 | {'✅ 已完成' if stats['stage2_success'] > 0 else '⏳ 待处理'} |
            | Stage 3 质量审查 | {stats['stage3_success']}/{stats['stage3_total']} 条 | {'✅ 已完成' if stats['stage3_success'] > 0 else '⏳ 待处理'} |
            | Stage 4 最终数据 | {stats['final_files']} 个文件 | {'✅ 已生成' if stats['final_files'] > 0 else '⏳ 待生成'} |
            """)

            refresh_btn = gr.Button("🔄 刷新统计", variant="secondary")
            refresh_stats = gr.Markdown()
            refresh_btn.click(
                fn=lambda: get_pipeline_stats(),
                outputs=refresh_stats,
            )

        with gr.Tab("📖 阶段详情"):
            for stage_key in ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]:
                with gr.Tab(stage_key):
                    gr.HTML(build_stage_detail_html(stage_key))

        with gr.Tab("🔍 数据查看"):
            gr.Markdown("### 查看各阶段输出数据示例")

            with gr.Row():
                stage_selector = gr.Dropdown(
                    choices=["Stage 1", "Stage 2", "Stage 3"],
                    value="Stage 1",
                    label="选择阶段",
                )
                record_index = gr.Number(value=0, label="记录索引", precision=0)
                view_btn = gr.Button("查看记录", variant="primary")

            record_display = gr.Textbox(
                label="数据记录内容",
                lines=20,
                interactive=False,
            )

            def view_record(stage: str, index: int) -> str:
                file_map = {
                    "Stage 1": DATA_DIRS["02_base_captions"] / "base_captions.jsonl",
                    "Stage 2": DATA_DIRS["03_dense_captions"] / "dense_captions.jsonl",
                    "Stage 3": DATA_DIRS["04_cleaned_metadata"] / "cleaned_metadata.jsonl",
                }
                record = load_sample_record(file_map[stage], int(index))
                return format_record_display(record)

            view_btn.click(
                fn=view_record,
                inputs=[stage_selector, record_index],
                outputs=record_display,
            )

        with gr.Tab("🛠️ 运行命令"):
            gr.Markdown("""
            ### 各阶段运行命令

            **Stage 1 - 基础标注:**
            ```bash
            python -m src.stage1_captioner
            python -m src.stage1_captioner --limit 10  # 测试模式
            ```

            **Stage 2 - 密集标注:**
            ```bash
            python -m src.stage2_dense_infer
            python -m src.stage2_dense_infer --limit 5 --keep-models-loaded  # 测试模式
            ```

            **部署裁判模型:**
            ```bash
            python deploy_judge_models.py --model qwen32b
            python deploy_judge_models.py --model llama70b
            python deploy_judge_models.py --model qwen35b-opus
            python deploy_judge_models.py --model all  # 全部下载
            ```

            **Stage 3 - 质量审查:**
            ```bash
            python -m src.stage3_reviewer --judge-model qwen32b
            python -m src.stage3_reviewer --judge-model llama70b --force-think
            python -m src.stage3_reviewer --limit 10  # 测试模式
            ```

            **本地双模型调试界面:**
            ```bash
            python app.py  # 启动 Gradio 调试界面
            ```
            """)

        with gr.Tab("📁 文件结构"):
            gr.Markdown("""
            ### 项目目录结构

            ```
            caption-generation-qc-pipeline/
            ├── data/
            │   ├── 01_raw_audio/           # 📥 原始音频
            │   ├── 02_base_captions/       # 📄 Stage 1 输出
            │   ├── 03_dense_captions/      # 📄 Stage 2 输出
            │   ├── 04_cleaned_metadata/    # 📄 Stage 3 输出
            │   └── 05_final_datasets/      # 📦 Stage 4 最终数据
            │       ├── sqa_cot/            # 单轮共情 CoT
            │       ├── aqa_cot/            # 音频问答 CoT
            │       └── multi_turn/         # 多轮对话
            ├── models/
            │   ├── qwen3-omni-captioner/   # Qwen 音频模型
            │   ├── af-next-captioner/      # AF-Next 音频模型
            │   └── judge/                  # 裁判模型目录
            │       ├── deepseek-r1-qwen-32b/
            │       ├── deepseek-r1-llama-70b/
            │       └── qwen35-opus-35b/
            ├── src/
            │   ├── stage1_captioner.py     # Stage 1 实现
            │   ├── stage2_dense_infer.py   # Stage 2 实现
            │   ├── stage3_reviewer.py      # Stage 3 实现
            │   ├── stage4_generator.py     # Stage 4 实现
            │   ├── main_pipeline.py        # 主调度入口
            │   └── utils/                  # 工具模块
            ├── app.py                      # 双模型调试界面
            ├── pipeline_visualizer.py      # 流程可视化界面
            ├── deploy_models.py            # 部署音频模型
            ├── deploy_judge_models.py      # 部署裁判模型
            └── requirements.txt            # 依赖列表
            ```
            """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)