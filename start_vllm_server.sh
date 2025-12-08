#!/bin/bash

# start_vllm_server.sh
# Wrapper script to start vLLM server with custom DeepSeek-OCR modes
#
# Usage:
#   ./start_vllm_server.sh [mode] [additional_vllm_args...]
#
# Modes:
#   tiny   - 512×512   (64 vision tokens)
#   small  - 640×640   (100 vision tokens)
#   base   - 1024×1024 (256 vision tokens)
#   large  - 1280×1280 (400 vision tokens)
#   gundam - 1024 base + 640 crops (dynamic)
#
# Examples:
#   ./start_vllm_server.sh large
#   ./start_vllm_server.sh base --gpu-memory-utilization 0.8
#   ./start_vllm_server.sh gundam --tensor-parallel-size 2

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="${1:-base}"
VALID_MODES=("tiny" "small" "base" "large" "gundam")

# Check if mode is valid
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo -e "${RED}Error: Invalid mode '${MODE}'${NC}"
    echo "Valid modes: ${VALID_MODES[@]}"
    exit 1
fi

# Shift to get additional arguments
shift || true

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepSeek-OCR vLLM Server Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Mode: ${MODE}${NC}"

# Apply monkeypatch
echo ""
echo -e "${YELLOW}Applying monkeypatch for mode: ${MODE}${NC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MONKEYPATCH_SCRIPT="${SCRIPT_DIR}/monkeypatch_vllm.py"

if [ ! -f "$MONKEYPATCH_SCRIPT" ]; then
    echo -e "${RED}Error: monkeypatch_vllm.py not found at ${MONKEYPATCH_SCRIPT}${NC}"
    exit 1
fi

python3 "$MONKEYPATCH_SCRIPT" --mode "$MODE"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Monkeypatch failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Monkeypatch applied successfully!${NC}"
echo ""

# Default vLLM arguments
MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-OCR}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
API_KEY="${API_KEY:-123}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-100}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1280}"

# Build vLLM command
VLLM_CMD="vllm serve \"${MODEL_NAME}\" \
    --host ${HOST} \
    --port ${PORT} \
    --api-key ${API_KEY} \
    --logits-processors \"vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor\" \
    --no-enable-prefix-caching \
    --mm-processor-cache-gb 0 \
    --calculate-kv-scales \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTIL} \
    --max-log-len 100000 \
    --disable-log-requests \
    --chat-template-content-format string \
    --disable-custom-all-reduce \
    --served-model-name \"ocr\" \
    --dtype bfloat16 \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --enable-chunked-prefill \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-model-len ${MAX_MODEL_LEN}"

# Add any additional arguments passed to the script
if [ $# -gt 0 ]; then
    VLLM_CMD="${VLLM_CMD} $@"
fi

echo -e "${YELLOW}Starting vLLM server...${NC}"
echo ""
echo -e "${BLUE}Command:${NC}"
echo "$VLLM_CMD"
echo ""
echo -e "${GREEN}Server will be available at: http://${HOST}:${PORT}${NC}"
echo -e "${GREEN}API Key: ${API_KEY}${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Execute vLLM command
eval $VLLM_CMD
