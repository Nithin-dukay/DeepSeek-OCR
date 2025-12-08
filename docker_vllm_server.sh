#!/bin/bash

# docker_vllm_server.sh
# Docker wrapper script to start vLLM server with custom DeepSeek-OCR modes
#
# Usage:
#   ./docker_vllm_server.sh [mode] [gpu_device] [port]
#
# Modes:
#   tiny   - 512×512   (64 vision tokens)
#   small  - 640×640   (100 vision tokens)
#   base   - 1024×1024 (256 vision tokens)
#   large  - 1280×1280 (400 vision tokens)
#   gundam - 1024 base + 640 crops (dynamic)
#
# Examples:
#   ./docker_vllm_server.sh large 0 8002
#   ./docker_vllm_server.sh base 1 8000
#   ./docker_vllm_server.sh gundam "0,1" 8003

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="${1:-base}"
GPU_DEVICE="${2:-0}"
PORT="${3:-8002}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm_ocr_${MODE}}"

VALID_MODES=("tiny" "small" "base" "large" "gundam")

# Check if mode is valid
if [[ ! " ${VALID_MODES[@]} " =~ " ${MODE} " ]]; then
    echo -e "${RED}Error: Invalid mode '${MODE}'${NC}"
    echo "Valid modes: ${VALID_MODES[@]}"
    exit 1
fi

# Mode configurations
declare -A MODE_BASE_SIZE
MODE_BASE_SIZE[tiny]=512
MODE_BASE_SIZE[small]=640
MODE_BASE_SIZE[base]=1024
MODE_BASE_SIZE[large]=1280
MODE_BASE_SIZE[gundam]=1024

declare -A MODE_IMAGE_SIZE
MODE_IMAGE_SIZE[tiny]=512
MODE_IMAGE_SIZE[small]=640
MODE_IMAGE_SIZE[base]=1024
MODE_IMAGE_SIZE[large]=1280
MODE_IMAGE_SIZE[gundam]=640

declare -A MODE_CROP_MODE
MODE_CROP_MODE[tiny]="False"
MODE_CROP_MODE[small]="False"
MODE_CROP_MODE[base]="False"
MODE_CROP_MODE[large]="False"
MODE_CROP_MODE[gundam]="True"

BASE_SIZE=${MODE_BASE_SIZE[$MODE]}
IMAGE_SIZE=${MODE_IMAGE_SIZE[$MODE]}
CROP_MODE=${MODE_CROP_MODE[$MODE]}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepSeek-OCR vLLM Docker Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Mode: ${MODE}${NC}"
echo -e "${GREEN}GPU Device: ${GPU_DEVICE}${NC}"
echo -e "${GREEN}Port: ${PORT}${NC}"
echo -e "${GREEN}Container: ${CONTAINER_NAME}${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  BASE_SIZE: ${BASE_SIZE}"
echo -e "  IMAGE_SIZE: ${IMAGE_SIZE}"
echo -e "  CROP_MODE: ${CROP_MODE}"
echo ""

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping and removing existing container: ${CONTAINER_NAME}${NC}"
    docker stop "${CONTAINER_NAME}" 2>/dev/null || true
    docker rm "${CONTAINER_NAME}" 2>/dev/null || true
fi

# Build the sed commands for patching
PATCH_COMMANDS="echo '=== Applying mode ${MODE} (${BASE_SIZE}x${BASE_SIZE}) ==='; \
PROC_FILE=\$(find /usr/local/lib -name 'deepseek_ocr.py' -path '*/transformers_utils/processors/*' -o -path '*/vllm/transformers_utils/processors/*' | head -1); \
MODEL_FILE=\$(find /usr/local/lib -name 'deepseek_ocr.py' -path '*/model_executor/models/*' | head -1); \
echo \"--- Processor file: \$PROC_FILE ---\"; \
echo \"--- Model file: \$MODEL_FILE ---\"; \
if [ -n \"\$PROC_FILE\" ]; then \
  echo '--- Modifying processor ---'; \
  sed -i 's/BASE_SIZE = [0-9]\\+/BASE_SIZE = ${BASE_SIZE}/' \"\$PROC_FILE\"; \
  sed -i 's/IMAGE_SIZE = [0-9]\\+/IMAGE_SIZE = ${IMAGE_SIZE}/' \"\$PROC_FILE\"; \
  sed -i 's/CROP_MODE = \\(True\\|False\\)/CROP_MODE = ${CROP_MODE}/' \"\$PROC_FILE\"; \
  echo '--- Verifying processor changes ---'; \
  grep -E '^(BASE_SIZE|IMAGE_SIZE|CROP_MODE)' \"\$PROC_FILE\" || echo 'Pattern not found at line start, checking anywhere...'; \
  grep -E '(BASE_SIZE|IMAGE_SIZE|CROP_MODE)' \"\$PROC_FILE\"; \
else \
  echo 'ERROR: Processor file not found'; \
fi; \
if [ -n \"\$MODEL_FILE\" ]; then \
  echo '--- Modifying model (use BASE_SIZE instead of vision_config) ---'; \
  sed -i 's/base_size = self\\.vision_config\\.image_size/base_size = BASE_SIZE/' \"\$MODEL_FILE\"; \
  echo '--- Verifying model changes ---'; \
  grep -n 'base_size = ' \"\$MODEL_FILE\" || echo 'No base_size assignment found'; \
else \
  echo 'WARNING: Model file not found (may not be needed)'; \
fi; \
echo '=== Starting vLLM ==='"

# Docker run command
echo -e "${YELLOW}Starting Docker container...${NC}"
echo ""

docker run --rm -d \
  --gpus "\"device=${GPU_DEVICE}\"" \
  --runtime=nvidia \
  --entrypoint /bin/bash \
  -e VLLM_USE_FLASHINFER_MOE_FP16=1 \
  -e TZ=UTC \
  -e HF_HUB_ENABLE_HF_TRANSFER="true" \
  -e RAY_DEDUP_LOGS=1 \
  -e VLLM_FLASH_ATTN_VERSION=2 \
  -p "${PORT}:8000" \
  -v "$(pwd)/data_aphro/cache:/app/aphrodite-engine/.cache" \
  -v "$(pwd)/data_aphro:/home/workspace" \
  -v "$(pwd)/datafolder:/data" \
  --name "${CONTAINER_NAME}" \
  vllm/vllm-openai:latest \
  -c "${PATCH_COMMANDS}; \
      vllm serve \"deepseek-ai/DeepSeek-OCR\" \
        --logits-processors \"vllm.model_executor.models.deepseek_ocr:NGramPerReqLogitsProcessor\" \
        --no-enable-prefix-caching --mm-processor-cache-gb 0 --calculate-kv-scales \
        --tensor-parallel-size 1 --api-key 123 \
        --gpu-memory-utilization 0.9 --max-log-len 100000 --disable-log-requests \
        --chat-template-content-format string --disable-custom-all-reduce \
        --served-model-name \"ocr\" \
        --dtype bfloat16 --max-num-seqs 100 --enable-chunked-prefill --max-num-batched-tokens 1280 \
        --max-model-len 8192"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Container started successfully!${NC}"
    echo ""
    echo -e "${BLUE}Container name: ${CONTAINER_NAME}${NC}"
    echo -e "${BLUE}Server URL: http://localhost:${PORT}${NC}"
    echo -e "${BLUE}API Key: 123${NC}"
    echo ""
    echo -e "${YELLOW}Useful commands:${NC}"
    echo -e "  View logs:    docker logs -f ${CONTAINER_NAME}"
    echo -e "  Stop server:  docker stop ${CONTAINER_NAME}"
    echo -e "  Remove:       docker rm ${CONTAINER_NAME}"
    echo ""
    echo -e "${YELLOW}Waiting for server to start (checking logs)...${NC}"
    sleep 3
    docker logs "${CONTAINER_NAME}"
else
    echo -e "${RED}✗ Failed to start container${NC}"
    exit 1
fi
