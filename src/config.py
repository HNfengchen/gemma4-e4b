from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "models"
LLAMA_BIN_DIR = BASE_DIR / "llama-cpp-bin"
SKILLS_DIR = BASE_DIR / "skills"

MODEL_FILE = MODEL_DIR / "Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf"
MMPROJ_FILE = MODEL_DIR / "mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"
LLAMA_SERVER = LLAMA_BIN_DIR / "llama-server.exe"
LLAMA_CLI = LLAMA_BIN_DIR / "llama-cli.exe"
AGENTS_MD = BASE_DIR / "AGENTS.md"
MCP_CONFIG = BASE_DIR / "mcp_config.json"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
LLAMA_SERVER_PORT = 8081

N_CTX = 8192
N_BATCH = 512
N_PARALLEL = 1

TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64

TOOL_CALL_MAX_ITERATIONS = 10
