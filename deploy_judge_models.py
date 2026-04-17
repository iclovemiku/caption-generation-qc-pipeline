"""
部署 Stage 3 裁判模型：用于数据校验和一致性审查。

包含三个推理蒸馏模型：
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-70B
- Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled

用法:
    python deploy_judge_models.py
    python deploy_judge_models.py --model qwen32b
    python deploy_judge_models.py --model llama70b
    python deploy_judge_models.py --model qwen35b-opus
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
JUDGE_MODEL_ROOT = PROJECT_ROOT / "models" / "judge"

JUDGE_MODELS = {
    "qwen32b": {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "local_dir": JUDGE_MODEL_ROOT / "deepseek-r1-qwen-32b",
        "description": "DeepSeek R1 蒸馏版，Qwen2.5-32B 基座，中文强",
    },
    "llama70b": {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        "local_dir": JUDGE_MODEL_ROOT / "deepseek-r1-llama-70b",
        "description": "DeepSeek R1 蒸馏版，Llama3.3-70B 基座，推理最强",
    },
    "qwen35b-opus": {
        "repo_id": "Jackrong/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled",
        "local_dir": JUDGE_MODEL_ROOT / "qwen35-opus-35b",
        "description": "Qwen3.5-35B MoE，Claude 4.6 Opus 推理蒸馏",
    },
}


def deploy_model(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deploy] {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[ok] {repo_id} -> {local_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="部署 Stage 3 裁判模型")
    parser.add_argument(
        "--model",
        choices=list(JUDGE_MODELS.keys()) + ["all"],
        default="all",
        help="选择要部署的模型，默认部署全部",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    JUDGE_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Stage 3 裁判模型部署")
    print("=" * 60)

    models_to_deploy = []
    if args.model == "all":
        models_to_deploy = list(JUDGE_MODELS.keys())
    else:
        models_to_deploy = [args.model]

    for model_key in models_to_deploy:
        info = JUDGE_MODELS[model_key]
        print(f"\n[{model_key}] {info['description']}")
        deploy_model(info["repo_id"], info["local_dir"])

    print("\n" + "=" * 60)
    print("[完成] 裁判模型部署结束")
    print("=" * 60)


if __name__ == "__main__":
    main()