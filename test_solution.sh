#!/bin/bash

# test_solution.sh
# Test script to verify the monkeypatch solution works correctly

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}DeepSeek-OCR MonkeyPatch Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASSED=0
FAILED=0

# Test function
test_command() {
    local test_name="$1"
    local command="$2"
    
    echo -e "${YELLOW}Testing: ${test_name}${NC}"
    
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
    echo ""
}

# Test 1: Check if scripts exist
echo -e "${BLUE}=== File Existence Tests ===${NC}"
echo ""

test_command "monkeypatch_vllm.py exists" "[ -f '${SCRIPT_DIR}/monkeypatch_vllm.py' ]"
test_command "start_vllm_server.sh exists" "[ -f '${SCRIPT_DIR}/start_vllm_server.sh' ]"
test_command "docker_vllm_server.sh exists" "[ -f '${SCRIPT_DIR}/docker_vllm_server.sh' ]"
test_command "example_client.py exists" "[ -f '${SCRIPT_DIR}/example_client.py' ]"
test_command "README_MONKEYPATCH.md exists" "[ -f '${SCRIPT_DIR}/README_MONKEYPATCH.md' ]"
test_command "QUICK_START.md exists" "[ -f '${SCRIPT_DIR}/QUICK_START.md' ]"

# Test 2: Check if scripts are executable
echo -e "${BLUE}=== Executable Tests ===${NC}"
echo ""

test_command "monkeypatch_vllm.py is executable" "[ -x '${SCRIPT_DIR}/monkeypatch_vllm.py' ]"
test_command "start_vllm_server.sh is executable" "[ -x '${SCRIPT_DIR}/start_vllm_server.sh' ]"
test_command "docker_vllm_server.sh is executable" "[ -x '${SCRIPT_DIR}/docker_vllm_server.sh' ]"
test_command "example_client.py is executable" "[ -x '${SCRIPT_DIR}/example_client.py' ]"

# Test 3: Check Python syntax
echo -e "${BLUE}=== Python Syntax Tests ===${NC}"
echo ""

test_command "monkeypatch_vllm.py syntax" "python3 -m py_compile '${SCRIPT_DIR}/monkeypatch_vllm.py'"
test_command "example_client.py syntax" "python3 -m py_compile '${SCRIPT_DIR}/example_client.py'"

# Test 4: Check script help output
echo -e "${BLUE}=== Help Output Tests ===${NC}"
echo ""

test_command "monkeypatch_vllm.py --help" "python3 '${SCRIPT_DIR}/monkeypatch_vllm.py' --help"
test_command "example_client.py --help" "python3 '${SCRIPT_DIR}/example_client.py' --help"

# Test 5: Check bash syntax
echo -e "${BLUE}=== Bash Syntax Tests ===${NC}"
echo ""

test_command "start_vllm_server.sh syntax" "bash -n '${SCRIPT_DIR}/start_vllm_server.sh'"
test_command "docker_vllm_server.sh syntax" "bash -n '${SCRIPT_DIR}/docker_vllm_server.sh'"

# Test 6: Check documentation
echo -e "${BLUE}=== Documentation Tests ===${NC}"
echo ""

test_command "README has mode descriptions" "grep -q 'Tiny.*Small.*Base.*Large.*Gundam' '${SCRIPT_DIR}/README_MONKEYPATCH.md'"
test_command "README has usage examples" "grep -q 'python monkeypatch_vllm.py' '${SCRIPT_DIR}/README_MONKEYPATCH.md'"
test_command "QUICK_START has commands" "grep -q './start_vllm_server.sh' '${SCRIPT_DIR}/QUICK_START.md'"

# Test 7: Check mode definitions
echo -e "${BLUE}=== Mode Definition Tests ===${NC}"
echo ""

test_command "Tiny mode defined" "grep -q 'tiny.*512' '${SCRIPT_DIR}/monkeypatch_vllm.py'"
test_command "Small mode defined" "grep -q 'small.*640' '${SCRIPT_DIR}/monkeypatch_vllm.py'"
test_command "Base mode defined" "grep -q 'base.*1024' '${SCRIPT_DIR}/monkeypatch_vllm.py'"
test_command "Large mode defined" "grep -q 'large.*1280' '${SCRIPT_DIR}/monkeypatch_vllm.py'"
test_command "Gundam mode defined" "grep -q 'gundam' '${SCRIPT_DIR}/monkeypatch_vllm.py'"

# Test 8: Check Docker script
echo -e "${BLUE}=== Docker Script Tests ===${NC}"
echo ""

test_command "Docker script has mode configs" "grep -q 'MODE_BASE_SIZE' '${SCRIPT_DIR}/docker_vllm_server.sh'"
test_command "Docker script has GPU support" "grep -q 'gpus' '${SCRIPT_DIR}/docker_vllm_server.sh'"
test_command "Docker script has vLLM command" "grep -q 'vllm serve' '${SCRIPT_DIR}/docker_vllm_server.sh'"

# Test 9: Check example client
echo -e "${BLUE}=== Example Client Tests ===${NC}"
echo ""

test_command "Client has OpenAI import" "grep -q 'from openai import OpenAI' '${SCRIPT_DIR}/example_client.py'"
test_command "Client has image encoding" "grep -q 'encode_image' '${SCRIPT_DIR}/example_client.py'"
test_command "Client has OCR function" "grep -q 'def ocr_document' '${SCRIPT_DIR}/example_client.py'"

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Passed: ${PASSED}${NC}"
echo -e "${RED}Failed: ${FAILED}${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Install vLLM: pip install vllm>=0.8.5"
    echo "2. Apply patch: python monkeypatch_vllm.py --mode large"
    echo "3. Start server: ./start_vllm_server.sh large"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    exit 1
fi
