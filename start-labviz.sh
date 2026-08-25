#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
api_root="$repo_root/V2.0/api"
web_root="$repo_root/V2.0/web"
venv_root="$api_root/.venv"
venv_python="$venv_root/bin/python"
refresh_dependencies=0
api_port="${LABVIZ_API_PORT:-8000}"
web_port="${LABVIZ_WEB_PORT:-3000}"

if [[ "${1:-}" == "--refresh-dependencies" ]]; then
  refresh_dependencies=1
fi

python_command="${PYTHON:-python3}"
if ! command -v "$python_command" >/dev/null 2>&1; then
  echo "Python 3.12 or 3.13 is required." >&2
  exit 1
fi
if ! "$python_command" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)'; then
  echo "Python 3.12 or 3.13 is required." >&2
  exit 1
fi
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Node.js 22.22.2 or newer is required." >&2
  exit 1
fi
if ! node -e "const [a,b,c]=process.versions.node.split('.').map(Number); process.exit(a>22||(a===22&&(b>22||(b===22&&c>=2)))?0:1)"; then
  echo "Node.js 22.22.2 or newer is required." >&2
  exit 1
fi

if [[ ! -x "$venv_python" ]]; then
  echo "Creating the LabViz Python environment..."
  "$python_command" -m venv "$venv_root"
  refresh_dependencies=1
fi

if [[ "$refresh_dependencies" -eq 1 ]]; then
  echo "Installing API dependencies..."
  "$venv_python" -m pip install --upgrade pip
  "$venv_python" -m pip install -r "$api_root/requirements.txt"
fi

if [[ "$refresh_dependencies" -eq 1 || ! -d "$web_root/node_modules" ]]; then
  echo "Installing website dependencies..."
  (cd "$web_root" && npm ci)
fi

api_pid=""
web_pid=""
cleanup() {
  [[ -n "$api_pid" ]] && kill "$api_pid" 2>/dev/null || true
  [[ -n "$web_pid" ]] && kill "$web_pid" 2>/dev/null || true
  wait "$api_pid" "$web_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

export LABVIZ_API_PROXY_TARGET="http://127.0.0.1:$api_port"

echo "Starting LabViz V2.0 on http://127.0.0.1:$web_port"
echo "Keep this terminal open. Local sign-in codes appear in the API output."
echo "Press Ctrl+C to stop both services."

(cd "$api_root" && exec "$venv_python" -m uvicorn labviz_api.main:app --host 127.0.0.1 --port "$api_port") &
api_pid=$!
(cd "$web_root" && exec npm run dev -- --hostname 127.0.0.1 --port "$web_port") &
web_pid=$!

while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
  sleep 1
done

echo "A LabViz service stopped unexpectedly." >&2
exit 1
