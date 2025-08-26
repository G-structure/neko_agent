{
  description = "A development environment for the Neko Agent project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ml-pkgs.url = "github:nixvital/ml-pkgs";
  };

  outputs = { self, nixpkgs, ml-pkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];

      # Common overlays used throughout
      nekoOverlays = [
        (import ./overlays/pylibsrtp.nix)
        (import ./overlays/aioice.nix)
        (import ./overlays/aiortc.nix)
        (import ./overlays/streaming.nix)
        (import ./overlays/cached-path.nix)
        (import ./overlays/ema-pytorch.nix)
        (import ./overlays/vocos.nix)
        (import ./overlays/transformers-stream-generator.nix)
        (import ./overlays/bitsandbytes.nix)
        (import ./overlays/f5-tts.nix)
        (import ./overlays/znver2-flags.nix)   # provides pkgs.nekoZnver2Env
        (import ./overlays/vmm-cli.nix)        # provides pkgs.vmm-cli
        ml-pkgs.overlays.torch-family
      ];

      # Helper function to get common system packages
      mkCommonSystemPackages = pkgs: with pkgs; [
        ffmpeg
        pkgconf
        libvpx
        just
        git
        curl
        wget
      ];

      devShellsBySystem = nixpkgs.lib.genAttrs supportedSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            config.cudaSupport = true;
            overlays = nekoOverlays;
          };

          # CUDA pin used in GPU shells/images
          cuda = pkgs.cudaPackages_12_8;

          # Common system packages used across shells
          commonSystemPackages = mkCommonSystemPackages pkgs;

          # Helper: load .env if present
          loadDotenv = ''
            if [ -f .env ]; then
              set -a; source .env; set +a
            fi
          '';

          # Common NPM AI tools setup used in multiple shells
          npmAITools = ''
            export NPM_CONFIG_PREFIX=$PWD/.npm-global
            export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
            if [ ! -x "$NPM_CONFIG_PREFIX/bin/codex" ]; then
              echo "Installing OpenAI Codex CLI..."
              npm install -g @openai/codex
            fi
            if [ ! -x "$NPM_CONFIG_PREFIX/bin/claude" ]; then
              echo "Installing Anthropic Claude CLI..."
              npm install -g @anthropic-ai/claude-code
            fi
          '';

          # CUDA env used by GPU shells
          cudaEnv = ''
            export CUDA_HOME=${cuda.cudatoolkit}
            export CUDA_PATH=$CUDA_HOME
            export PATH=$CUDA_HOME/bin:$PATH
            export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${cuda.cudnn}/lib:$LD_LIBRARY_PATH
            export NVIDIA_VISIBLE_DEVICES=''${NVIDIA_VISIBLE_DEVICES:-all}
            export NVIDIA_DRIVER_CAPABILITIES=''${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}
            export CUDA_MODULE_LOADING=''${CUDA_MODULE_LOADING:-LAZY}
            export PYTORCH_CUDA_ALLOC_CONF=''${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
          '';

          # Common Python packages used across environments
          commonPythonPackages = ps: [
            ps.transformers
            ps.pillow
            ps.websockets
            ps."prometheus-client"
            ps.av
            pkgs.python3Packages.pylibsrtp
            pkgs.python3Packages.aioice
            pkgs.python3Packages.aiortc
            pkgs.python3Packages.streaming
            pkgs.python3Packages.f5-tts
            ps.numpy
            ps.scipy
            ps.requests
            ps.setuptools
            ps.wheel
            ps.zstandard
            ps.xxhash
            ps.tqdm
          ];

          # Portable Python env (CUDA wheels)
          pyEnvGeneric = pkgs.python3.withPackages (ps: [
            ps."torch-bin"
            ps."torchvision-bin"
            ps."torchaudio-bin"
            (ps.accelerate.override { torch = ps."torch-bin"; })
          ] ++ commonPythonPackages ps);

          # Swap to from-source PyTorch here if/when you want to build locally.
          pyEnvOpt = pyEnvGeneric;

          # Zen2 flags script (Bash env exports) from overlay
          znver2File = pkgs.nekoZnver2Env;

        in
        {
          default = pkgs.mkShell {
            buildInputs = commonSystemPackages ++ [
              pkgs.python3Packages.pylibsrtp
              pkgs.python3Packages.aioice
              pkgs.python3Packages.aiortc
              pkgs.python3Packages.streaming
              pkgs.python3Packages.f5-tts
              (pkgs.python3.withPackages (ps: with ps; [
                transformers
                torch
                torchvision
                pillow
                accelerate
                websockets
                prometheus-client
                av
                zstandard
                xxhash
                tqdm
              ]))
              pkgs.nodejs_20
              pkgs.nodePackages.npm
            ];
            shellHook = ''
              ${loadDotenv}
              # Quick sanity check so users see it's present
              python -c "import importlib.util; m='streaming'; print(f'[{m}]', 'OK' if importlib.util.find_spec(m) else 'MISSING')" || true
              ${npmAITools}
            '';
          };

          gpu = pkgs.mkShell {
            packages = commonSystemPackages ++ [
              cuda.cudatoolkit
              cuda.cudnn
              cuda.nccl
              pyEnvGeneric
              pkgs.nodejs_20
              pkgs.nodePackages.npm
            ];
            shellHook = ''
              ${loadDotenv}
              ${cudaEnv}
              python -c "import importlib.util; m='streaming'; print(f'[{m}]', 'OK' if importlib.util.find_spec(m) else 'MISSING')" || true
              python - <<'PY'
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {getattr(torch.version, 'cuda', None)}, avail={torch.cuda.is_available()}")
except Exception as e:
    print("Torch check:", e)
PY
              ${npmAITools}
            '';
          };

          ai = pkgs.mkShell {
            packages = commonSystemPackages ++ [
              pkgs.nodejs_20
              pkgs.nodePackages.npm
            ];
            shellHook = ''
              ${loadDotenv}
              ${npmAITools}
            '';
          };

          neko = pkgs.mkShell {
            buildInputs = [
              pkgs.colima
              pkgs.docker
              pkgs.docker-buildx
              pkgs.docker-compose
              pkgs.curl
              pkgs.jq
              pkgs.nodejs_20
              pkgs.nodePackages.npm
              (pkgs.writeShellScriptBin "neko-services" ''
                COMPOSE_FILE="${./docker-compose.yml}"
                ensure_colima() {
                  if ! ${pkgs.colima}/bin/colima status >/dev/null 2>&1; then
                    echo "Starting Colima..."
                    ${pkgs.colima}/bin/colima start --vm-type vz --cpu 2 --memory 4 --mount-type sshfs --mount "~:w"
                    sleep 5
                  fi
                  export DOCKER_HOST="unix://$(readlink -f ~/.colima/default/docker.sock)"
                }
                case "$1" in
                  up) echo "Starting Neko..."; ensure_colima; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d ;;
                  down) ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" down ;;
                  restart) ensure_colima; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" restart ;;
                  pull) ensure_colima; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull ;;
                  logs) ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" logs -f ''${2:-} ;;
                  ps) ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps ;;
                  status) if ${pkgs.colima}/bin/colima status >/dev/null 2>&1; then ${pkgs.colima}/bin/colima status; else echo "Colima not running"; fi; ensure_colima; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps ;;
                  update) ensure_colima; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull; ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d ;;
                  run-docker) ensure_colima; bash "$COMPOSE_FILE" ;;
                  *) echo "Usage: neko-services {up|down|restart|pull|logs|ps|status|update|run-docker}"; exit 1 ;;
                esac
              '')
            ];
            shellHook = ''
              echo "Neko Docker environment loaded!"
              echo "Neko at http://localhost:8080"
              export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
              ${npmAITools}
            '';
          };

          cpu-opt = pkgs.mkShell {
            buildInputs = commonSystemPackages ++ [ pyEnvOpt ];
            shellHook = ''
              ${loadDotenv}
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux ''
                source ${znver2File}
                echo "[cpu-opt] Using znver2 flags: $NIX_CFLAGS_COMPILE"
              ''}
            '';
          };

          gpu-opt = pkgs.mkShell {
            packages = commonSystemPackages ++ [
              cuda.cudatoolkit
              cuda.cudnn cuda.nccl
              pyEnvOpt
            ];
            shellHook = ''
              ${loadDotenv}
              ${pkgs.lib.optionalString pkgs.stdenv.isLinux "source ${znver2File}"}
              ${cudaEnv}
              export TORCH_CUDA_ARCH_LIST=''${TORCH_CUDA_ARCH_LIST:-8.6}
              echo "[gpu-opt] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
            '';
          };

          docs = pkgs.mkShell {
            buildInputs = commonSystemPackages ++ [
              pkgs.mdbook
              pkgs.mdbook-mermaid
              pkgs.mdbook-linkcheck
              pkgs.mdbook-toc
              pkgs.nodejs_20
              pkgs.nodePackages.npm
              (pkgs.python3.withPackages (ps: with ps; [
                sphinx
                sphinx-rtd-theme
                myst-parser
              ]))
            ];
            shellHook = ''
              ${loadDotenv}
              echo "📚 Neko Agent Documentation Environment"
              echo "Commands:"
              echo "  mdbook serve --open   - Start dev server (from docs/)"
              echo "  mdbook build          - Build static docs"
              echo "  mdbook test           - Test documentation"
              echo ""
              ${npmAITools}
            '';
          };

          tee = pkgs.mkShell {
            buildInputs = commonSystemPackages ++ [
              pkgs.nodejs_20
              pkgs.nodePackages.npm
              pkgs.docker
              pkgs.docker-compose
              pkgs.bun
              pkgs.vmm-cli
            ];
            shellHook = ''
              ${loadDotenv}
              echo "🔐 TEE Development Environment"
              echo "Tools available:"
              echo "  - Phala Cloud CLI (phala)"
              echo "  - Legacy VMM CLI (vmm-cli)"
              echo "  - Docker & Docker Compose"
              echo "  - Bun runtime"
              echo ""
              
              # Setup Phala CLI
              export NPM_CONFIG_PREFIX=$PWD/.npm-global
              export PATH=$NPM_CONFIG_PREFIX/bin:$PATH
              
              if [ ! -x "$NPM_CONFIG_PREFIX/bin/phala" ]; then
                echo "Installing Phala Cloud CLI..."
                npm install -g phala
              fi
              
              echo "Phala CLI version: $(phala --version 2>/dev/null || echo 'Installing...')"
              echo "Legacy VMM CLI: $(vmm-cli --version 2>/dev/null || echo 'Available')"
              echo ""
              echo "Quick start (Modern CLI):"
              echo "  phala auth login <your-api-key>    - Login to Phala Cloud"
              echo "  phala status                       - Check authentication status"
              echo "  phala cvms list                    - List your CVMs"
              echo "  phala nodes                        - List available TEE nodes"
              echo ""
              echo "Quick start (Legacy VMM CLI):"
              echo "  export DSTACK_VMM_URL=https://web.h200-dstack-01.phala.network:40081/"
              echo "  vmm-cli lsvm                       - List virtual machines"
              echo "  vmm-cli lsimage                    - List available images"
              echo "  vmm-cli lsgpu                      - List available GPUs"
              echo ""
            '';
          };
        }
      );
    in
    {
      # Dev shells
      devShells = devShellsBySystem;

      # ----------------------------
      # Docker images (x86_64-linux)
      # ----------------------------
      packages.x86_64-linux =
        let
          pkgs = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            config.cudaSupport = true;
            overlays = nekoOverlays;
          };
          cuda = pkgs.cudaPackages_12_8;

          # Reuse the same python environment definitions from devShells
          pyEnvGeneric = pkgs.python3.withPackages (ps: [
            ps."torch-bin"
            ps."torchvision-bin"
            ps."torchaudio-bin"
            (ps.accelerate.override { torch = ps."torch-bin"; })
          ] ++ (let commonPythonPackages = ps: [
            ps.transformers
            ps.pillow
            ps.websockets
            ps."prometheus-client"
            ps.av
            pkgs.python3Packages.pylibsrtp
            pkgs.python3Packages.aioice
            pkgs.python3Packages.aiortc
            pkgs.python3Packages.streaming
            pkgs.python3Packages.f5-tts
            ps.numpy
            ps.scipy
            ps.requests
            ps.setuptools
            ps.wheel
            ps.zstandard
            ps.xxhash
            ps.tqdm
          ]; in commonPythonPackages ps));

          pyEnvOpt = pyEnvGeneric;

          # Small runner that executes your agent
          runnerGeneric = pkgs.writeShellScriptBin "neko-agent" ''
            exec ${pyEnvGeneric}/bin/python ${./src/agent.py} "$@"
          '';
          runnerOpt = pkgs.writeShellScriptBin "neko-agent" ''
            exec ${pyEnvOpt}/bin/python ${./src/agent.py} "$@"
          '';

          baseEnv = [
            "PYTHONUNBUFFERED=1"
            "NEKO_LOGLEVEL=INFO"
          ];

          # helper to assemble a root FS with /bin symlinks (replaces deprecated `contents`)
          mkRoot = paths: pkgs.buildEnv {
            name = "image-root";
            inherit paths;
            pathsToLink = [ "/bin" ];
          };
        in
        {
          # Portable CUDA image (includes PTX; works beyond 8.6)
          neko-agent-docker-generic = pkgs.dockerTools.buildImage {
            name = "neko-agent:cuda12.8-generic";
            created = "now";
            copyToRoot = mkRoot ([
              runnerGeneric
              pyEnvGeneric
              cuda.cudatoolkit
              cuda.cudnn
              cuda.nccl
              pkgs.bashInteractive
            ] ++ mkCommonSystemPackages pkgs);
            config = {
              Env = baseEnv ++ [
                "CUDA_HOME=${cuda.cudatoolkit}"
                "LD_LIBRARY_PATH=${cuda.cudatoolkit}/lib64:${cuda.cudnn}/lib"
                "CUDA_MODULE_LOADING=LAZY"
                "TORCH_CUDA_ARCH_LIST=8.6+PTX"
              ];
              WorkingDir = "/workspace";
              Entrypoint = [ "/bin/neko-agent" ];
            };
          };

          # Optimized CUDA image (sm_86 only + znver2 CPU flags)
          neko-agent-docker-opt = pkgs.dockerTools.buildImage {
            name = "neko-agent:cuda12.8-sm86-v3";
            created = "now";
            copyToRoot = mkRoot ([
              runnerOpt
              pyEnvOpt
              cuda.cudatoolkit
              cuda.cudnn cuda.nccl
              pkgs.bashInteractive
              (pkgs.writeShellScriptBin "neko-znver2-env" "source ${pkgs.nekoZnver2Env}; exec \"$@\"")
            ] ++ mkCommonSystemPackages pkgs);
            config = {
              Env = baseEnv ++ [
                "CUDA_HOME=${cuda.cudatoolkit}"
                "LD_LIBRARY_PATH=${cuda.cudatoolkit}/lib64:${cuda.cudnn}/lib"
                "CUDA_MODULE_LOADING=LAZY"
                "TORCH_CUDA_ARCH_LIST=8.6"
                "NIX_CFLAGS_COMPILE=-O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
                "RUSTFLAGS=-C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
              ];
              WorkingDir = "/workspace";
              Entrypoint = [ "/bin/neko-agent" ];
            };
          };
        };

      # helper to build both images
      apps.x86_64-linux.build-images =
        let
          pkgs-app = import nixpkgs { system = "x86_64-linux"; };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "build-images" ''
            set -euo pipefail
            nix build .#neko-agent-docker-generic
            nix build .#neko-agent-docker-opt
            echo "Images built."
          '');
        };

      # Documentation apps
      apps.x86_64-linux.docs-build =
        let
          pkgs-app = import nixpkgs { system = "x86_64-linux"; };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "docs-build" ''
            set -euo pipefail
            cd docs
            export PATH=${pkgs-app.mdbook}/bin:${pkgs-app.mdbook-mermaid}/bin:${pkgs-app.mdbook-linkcheck}/bin:${pkgs-app.mdbook-toc}/bin:$PATH
            ${pkgs-app.mdbook}/bin/mdbook build
            echo "📚 Documentation built in docs/book/"
          '');
        };

      apps.x86_64-linux.docs-serve =
        let
          pkgs-app = import nixpkgs { system = "x86_64-linux"; };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "docs-serve" ''
            set -euo pipefail
            cd docs
            echo "📚 Starting documentation server at http://localhost:3000"
            echo "📝 Edit files in docs/src/ for live reload"
            export PATH=${pkgs-app.mdbook}/bin:${pkgs-app.mdbook-mermaid}/bin:${pkgs-app.mdbook-linkcheck}/bin:${pkgs-app.mdbook-toc}/bin:$PATH
            ${pkgs-app.mdbook}/bin/mdbook serve --hostname 0.0.0.0 --port 3000 --open
          '');
        };

      apps.x86_64-linux.docs-check =
        let
          pkgs-app = import nixpkgs { system = "x86_64-linux"; };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "docs-check" ''
            set -euo pipefail
            cd docs
            echo "🔍 Checking documentation links and content..."
            export PATH=${pkgs-app.mdbook}/bin:${pkgs-app.mdbook-mermaid}/bin:${pkgs-app.mdbook-linkcheck}/bin:${pkgs-app.mdbook-toc}/bin:$PATH
            ${pkgs-app.mdbook}/bin/mdbook test
            echo "✅ Documentation validation complete"
          '');
        };
    };
}
