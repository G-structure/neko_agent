# Getting Started

This project loads environment variables from a local `.env` file when you enter the Nix dev shell. **Never commit your real `.env` file**—only the template `.env.example` should be tracked.

## 1. Prepare your environment variables

```bash
cp .env.example .env
# Edit `.env` and set NEKO_URL, NEKO_USER, NEKO_PASS, or NEKO_WS, etc.
echo ".env" >> .gitignore

	•	Edit .env with your secret/project-specific values.
	•	All env vars must be in KEY=value format (no export).
	•	You can quote values containing spaces or JSON.

2. Enter the Nix flake development shell

nix develop

The dev shell automatically loads all variables from .env.
No need to modify the flake.nix to add secrets.

3. Run the agent

REST login flow (recommended for password-auth):

python src/agent.py \
  --neko-url "$NEKO_URL" \
  --username "$NEKO_USER" \
  --password "$NEKO_PASS"

Direct WebSocket flow (token in NEKO_WS):

# Set NEKO_WS in .env
python src/agent.py

Optional flags:
	•	--mode web|phone (or NEKO_MODE in .env)
	•	--max-steps N (or NEKO_MAX_STEPS)
	•	--metrics-port PORT (or NEKO_METRICS_PORT)
	•	--loglevel DEBUG|INFO|... (or NEKO_LOGLEVEL)
	•	--no-audio to disable audio (NEKO_AUDIO=0 in .env)

4. Notes and tips
	•	Frame requests default to 1280x720; to use 1280x800, adjust the agent source.
	•	On macOS/MPS, OFFLOAD_FOLDER will be created automatically (ensure you have disk space).
	•	Set CLICK_SAVE_PATH to save all action-marked frames, or FRAME_SAVE_PATH for single-frame dumps.
	•	NEKO_LOGLEVEL=DEBUG gives verbose logs.
	•	Metrics available at: http://localhost:${NEKO_METRICS_PORT} (default: 9000).

5. Troubleshooting
	•	Ensure .env is present in the project root, with all needed variables set.
	•	Use NEKO_LOGLEVEL=DEBUG in .env for troubleshooting.
	•	If agent.py complains about missing envs, double-check .env and shell loading.
