#!/usr/bin/env bash
# 切换 pyproject 版本: pinned (老 Mac/torch 2.2.2) 或 latest (现代环境/torch >= 2.4)
# 用法: ./use-version.sh pinned    或    ./use-version.sh latest

set -e
cd "$(dirname "$0")"

case "$1" in
    pinned)
        cp pyproject.pinned.toml pyproject.toml
        echo "✓ 切换到 pinned 版本 (torch==2.2.2, 兼容 macOS x86_64)"
        ;;
    latest)
        cp pyproject.latest.toml pyproject.toml
        echo "✓ 切换到 latest 版本 (torch>=2.4, 现代 Linux/arm64 Mac/Windows)"
        ;;
    *)
        echo "用法: $0 {pinned|latest}"
        echo ""
        echo "  pinned  → 当前 macOS x86_64 必须用这个 (torch 2.2.2)"
        echo "  latest  → 其他机器推荐 (torch >= 2.4, 更快、新特性多)"
        exit 1
        ;;
esac

echo "下一步: rm -rf .venv uv.lock && uv sync --extra neural"
