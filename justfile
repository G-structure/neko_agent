# Neko Agent Justfile

# Start the nix development shell
shell:
    nix develop

# Start the nix GPU development shell (Ubuntu/NVIDIA)
gpu:
    NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- nix develop .#gpu

pydebug:
    NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- nix develop .#pydebug

# Launch the agent with a default task
agent:
    NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "search for pics of cats"
#    #!/usr/bin/env bash
#    cd "{{justfile_directory()}}"
#    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "search for pics of cats"'

# Launch the agent with a custom task
agent-task task:
    #!/usr/bin/env bash
    cd "{{justfile_directory()}}"
    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/agent.py --task "{{task}}"'

# Force kill the agent script
kill-agent:
    #!/usr/bin/env bash
    echo "Killing agent processes..."
    ps aux | grep -E "(src/agent|just agent)" | grep -v grep
    pkill -9 -f "python src/agent" || echo "No python agent process found"
    pkill -9 -f "just agent" || echo "No just agent process found"
    pkill -9 -f "src/agent.py" || echo "No agent.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/agent|just agent)" | grep -v grep || echo "No remaining agent processes"

# Launch the manual control script
manual:
    NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko_manual.log" python src/manual.py
#    #!/usr/bin/env bash
#    cd "{{justfile_directory()}}"
#    nix develop --command bash -c 'NEKO_LOGFILE="/Users/luc/projects/neko_agent/tmp/neko1.log" python src/manual.py'

# Launch the agent via uv with .env
uv-agent:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    if [ -f .env ]; then
      set -a
      source .env
      set +a
    fi
    uv run src/agent.py

# Force kill the manual control script
kill-manual:
    #!/usr/bin/env bash
    echo "Killing manual processes..."
    ps aux | grep -E "(src/manual|just manual)" | grep -v grep
    pkill -9 -f "python src/manual" || echo "No python manual process found"
    pkill -9 -f "just manual" || echo "No just manual process found"
    pkill -9 -f "src/manual.py" || echo "No manual.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/manual|just manual)" | grep -v grep || echo "No remaining manual processes"

# Force kill both agent and manual control scripts
kill-all:
    #!/usr/bin/env bash
    echo "Killing all neko processes..."
    ps aux | grep -E "(src/agent|src/manual|just agent|just manual)" | grep -v grep
    pkill -9 -f "python src/agent" || echo "No python agent process found"
    pkill -9 -f "just agent" || echo "No just agent process found"
    pkill -9 -f "python src/manual" || echo "No python manual process found"
    pkill -9 -f "just manual" || echo "No just manual process found"
    pkill -9 -f "src/agent.py" || echo "No agent.py process found"
    pkill -9 -f "src/manual.py" || echo "No manual.py process found"
    sleep 1
    echo "Remaining processes:"
    ps aux | grep -E "(src/agent|src/manual|just agent|just manual)" | grep -v grep || echo "No remaining neko processes"

# Launch default browser at NEKO_URL
browser:
    open "$NEKO_URL"

# Create necessary directories
setup:
    mkdir -p tmp/actions

# View the agent log
log:
    tail -f /Users/luc/projects/neko_agent/tmp/neko_agent.log

# Show running neko processes
ps:
    ps aux | grep -E "(src/agent.py|src/manual.py)" | grep -v grep

# Clean up temporary files
clean:
    rm -rf tmp/actions/*
    rm -f tmp/frame.png
    rm -f tmp/neko1.log
    rm -f tmp/*.tmp

# === NEW: Optimized development shells ===

# CPU-optimized shell (znver2 flags)
cpu-opt:
    nix develop .#cpu-opt

# GPU-optimized shell (znver2 + CUDA sm_86)
gpu-opt:
    NIXPKGS_ALLOW_UNFREE=1 nix run --impure github:nix-community/nixGL#nixGLNvidia -- nix develop .#gpu-opt

# === NEW: Docker image management ===

# Build both Docker images
images:
    nix build .#neko-agent-docker-generic .#neko-agent-docker-opt

# Build generic Docker image (portable)
docker-build-generic:
    nix build .#neko-agent-docker-generic

# Build optimized Docker image (znver2 + sm_86)
docker-build-optimized:
    nix build .#neko-agent-docker-opt

# Run generic Docker image (requires NVIDIA Container Toolkit)
docker-run-generic:
    #!/usr/bin/env bash
    echo "Loading generic CUDA image..."
    docker load < result
    echo "Running neko-agent:cuda12.8-generic with GPU support..."
    docker run --rm --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e NEKO_WS="${NEKO_WS:-}" \
        -e NEKO_LOGLEVEL="${NEKO_LOGLEVEL:-INFO}" \
        neko-agent:cuda12.8-generic

# Run optimized Docker image (requires NVIDIA Container Toolkit)
docker-run-optimized:
    #!/usr/bin/env bash
    echo "Loading optimized CUDA image..."
    docker load < result
    echo "Running neko-agent:cuda12.8-sm86-v3 with GPU support..."
    docker run --rm --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
        -e NEKO_WS="${NEKO_WS:-}" \
        -e NEKO_LOGLEVEL="${NEKO_LOGLEVEL:-INFO}" \
        neko-agent:cuda12.8-sm86-v3

# Build all images using Nix app
build-images:
    nix run .#build-images

# === VMM GPU Management ===

# List available GPUs using VMM CLI
gpus:
    #!/usr/bin/env bash
    cd "{{justfile_directory()}}"
    nix develop .#tee --command bash -c 'vmm-cli lsgpu'

# Deploy application with VMM CLI
# Usage: just deploy docker-compose.yml "0a:00.0 1a:00.0"
# Usage: just deploy docker-compose.yml "0a:00.0"
deploy-gpu compose_file gpus:
    #!/usr/bin/env bash
    cd "{{justfile_directory()}}"
    compose_name=$(basename "{{compose_file}}" .yml | sed 's/\.yaml$//')
    nix develop .#tee --command bash -c '\
        vmm-cli compose \
            --name "'$compose_name'" \
            --docker-compose "{{compose_file}}" \
            --gateway \
            --kms \
            --public-logs \
            --public-sysinfo \
            --output ./neko-app-compose.json && \
        vmm-cli deploy \
            --name "'$compose_name'-${USER:-unknown}" \
            --image "dstack-nvidia-dev-0.5.3" \
            --compose ./neko-app-compose.json \
            --vcpu 12 \
            --memory 64G \
            --disk 100G \
            $(echo "{{gpus}}" | tr " " "\n" | sed "s/^/--gpu /" | tr "\n" " ")'

deploy compose_file:
    #!/usr/bin/env bash
    cd "{{justfile_directory()}}"
    compose_name=$(basename "{{compose_file}}" .yml | sed 's/\.yaml$//')
    nix develop .#tee --command bash -c '\
        vmm-cli compose \
            --name "'$compose_name'" \
            --docker-compose "{{compose_file}}" \
            --gateway \
            --kms \
            --public-logs \
            --public-sysinfo \
            --output ./temp-neko-app-compose.json && \
        vmm-cli deploy \
            --name "'$compose_name'-${USER:-unknown}" \
            --image "dstack-nvidia-dev-0.5.3" \
            --compose ./temp-neko-app-compose.json \
            --vcpu 2 \
            --memory 8G \
            --disk 10G'

# Connect to deployed VM via SSH using App ID
# Usage: just connect aa7d1365f90c90ac2be2ac6139237afab9154611
connect app_id:
    #!/usr/bin/env bash
    ssh -vvv -o HostName={{app_id}}-1022.h200-dstack-01.phala.network \
        -o "ProxyCommand=openssl s_client -servername %h -quiet -connect %h:11354" \
        -o ControlMaster=no \
        -o ControlPath=none \
        -p 22 root@my-tee-box

# === Finetuning ===

# Train on captured MDS episodes (uses .env)
train:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    if [ -f .env ]; then set -a; source .env; set +a; fi
    python src/train.py

# Train with uv runner (loads .env)
uv-train:
    #!/usr/bin/env bash
    set -euo pipefail
    cd "{{justfile_directory()}}"
    if [ -f .env ]; then set -a; source .env; set +a; fi
    uv run src/train.py

# Docker cleanup
docker-clean:
    #!/usr/bin/env bash
    docker image prune -a -f
    docker container prune -f
    docker builder prune -a -f
