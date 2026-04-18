from __future__ import annotations

import os
import argparse
import sys
import zipfile
import requests
from pathlib import Path

from src.config import (
    BASE_DIR,
    LLAMA_BIN_DIR,
    LLAMA_SERVER,
    MODEL_FILE,
    MMPROJ_FILE,
    SERVER_HOST,
    SERVER_PORT,
)

LLAMA_CPP_RELEASE = "b8839"
LLAMA_CPP_BIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/llama-{LLAMA_CPP_RELEASE}-bin-win-cuda-12.4-x64.zip"
CUDART_BIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_RELEASE}/cudart-llama-bin-win-cuda-12.4-x64.zip"

MODEL_REPO = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
MODEL_FILES = [
    "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf",
    "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
]


def download_file(url: str, target: str, desc: str = ""):
    if os.path.exists(target):
        size_mb = os.path.getsize(target) / (1024**2)
        print(f"[SKIP] {desc} already exists ({size_mb:.0f} MB)")
        return True
    print(f"[DOWNLOAD] {desc or url} ...")
    try:
        resp = requests.get(url, stream=True, timeout=30)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(target, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded/(1024**2):.1f}/{total/(1024**2):.1f} MB ({pct:.0f}%)", end="", flush=True)
        print(f"\n[DONE] {desc}")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {desc}: {e}")
        if os.path.exists(target):
            os.remove(target)
        return False


def download_llama_cpp():
    if LLAMA_SERVER.exists():
        print(f"[SKIP] llama.cpp binaries already exist")
        return True

    bin_zip = str(BASE_DIR / "llama-cpp-bin.zip")
    cudart_zip = str(BASE_DIR / "cudart-bin.zip")

    if not download_file(LLAMA_CPP_BIN_URL, bin_zip, "llama.cpp binaries"):
        return False
    if not download_file(CUDART_BIN_URL, cudart_zip, "CUDA runtime"):
        return False

    LLAMA_BIN_DIR.mkdir(parents=True, exist_ok=True)
    print("[EXTRACT] llama.cpp binaries...")
    with zipfile.ZipFile(bin_zip, "r") as z:
        z.extractall(str(LLAMA_BIN_DIR))
    print("[EXTRACT] CUDA runtime...")
    with zipfile.ZipFile(cudart_zip, "r") as z:
        z.extractall(str(LLAMA_BIN_DIR))

    for f in [bin_zip, cudart_zip]:
        if os.path.exists(f):
            os.remove(f)

    print("[DONE] llama.cpp setup complete")
    return True


def download_models():
    from huggingface_hub import hf_hub_download

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    for fname in MODEL_FILES:
        target = models_dir / fname
        if target.exists():
            size_mb = target.stat().st_size / (1024**2)
            print(f"[SKIP] {fname} already exists ({size_mb:.0f} MB)")
            continue
        print(f"[DOWNLOAD] {fname} ...")
        hf_hub_download(
            repo_id=MODEL_REPO,
            filename=fname,
            local_dir=str(models_dir),
        )
        size_mb = target.stat().st_size / (1024**2)
        print(f"[DONE] {fname} ({size_mb:.0f} MB)")

    return True


def show_info():
    print("System Information:")
    print(f"  llama-server: {LLAMA_SERVER} (exists: {LLAMA_SERVER.exists()})")
    print(f"  Model file: {MODEL_FILE} (exists: {MODEL_FILE.exists()})")
    if MODEL_FILE.exists():
        print(f"  Model size: {MODEL_FILE.stat().st_size / (1024**3):.2f} GB")
    print(f"  mmproj file: {MMPROJ_FILE} (exists: {MMPROJ_FILE.exists()})")
    if MMPROJ_FILE.exists():
        print(f"  mmproj size: {MMPROJ_FILE.stat().st_size / (1024**2):.1f} MB")

    try:
        import torch
        print(f"\n  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Gemma-4-E4B Deployment Suite")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("download", help="Download model files and llama.cpp binaries")
    subparsers.add_parser("chat", help="Start interactive chat")
    subparsers.add_parser("server", help="Start OpenAI-compatible API server")

    server_parser = subparsers.add_parser("serve", help="Start API server")
    server_parser.add_argument("--host", default=SERVER_HOST)
    server_parser.add_argument("--port", type=int, default=SERVER_PORT)

    subparsers.add_parser("info", help="Show system and model info")

    args = parser.parse_args()

    if args.command == "download":
        download_llama_cpp()
        download_models()

    elif args.command in ("server", "serve"):
        from src.server import run_server
        run_server(host=args.host, port=args.port)

    elif args.command == "chat":
        from chat import main as chat_main
        chat_main()

    elif args.command == "info":
        show_info()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
