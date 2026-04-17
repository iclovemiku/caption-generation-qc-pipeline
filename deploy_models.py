from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ROOT = PROJECT_ROOT / "models"

MODELS = {
    "Qwen/Qwen3-Omni-30B-A3B-Captioner": MODEL_ROOT / "qwen3-omni-captioner",
    "nvidia/audio-flamingo-next-captioner-hf": MODEL_ROOT / "af-next-captioner",
}


def deploy_model(repo_id: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[deploy] {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir))
    print(f"[ok] {repo_id}")


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    for repo_id, local_dir in MODELS.items():
        deploy_model(repo_id, local_dir)
    print("[done] all models deployed")


if __name__ == "__main__":
    main()
