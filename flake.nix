{
  description = "A development environment for the Neko Agent project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    ml-pkgs.url = "github:nixvital/ml-pkgs";
  };

  outputs = { self, nixpkgs, ml-pkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];

      # Build metadata for reproducible builds and attestation
      buildInfo = rec {
        # Use flake lock for reproducible timestamps, convert to RFC3339 format
        timestamp = if (self ? lastModifiedDate)
          then let
            raw = self.lastModifiedDate;
            year = builtins.substring 0 4 raw;
            month = builtins.substring 4 2 raw;
            day = builtins.substring 6 2 raw;
            hour = builtins.substring 8 2 raw;
            minute = builtins.substring 10 2 raw;
            second = builtins.substring 12 2 raw;
          in "${year}-${month}-${day}T${hour}:${minute}:${second}Z"
          else "1970-01-01T00:00:00Z"; # Unix epoch as fallback

        # Git information from flake
        revision = self.rev or self.dirtyRev or "unknown";
        shortRev = builtins.substring 0 8 revision;

        # Nix metadata
        nixpkgsRev = nixpkgs.rev or "unknown";

        # Version derivation
        version = if (self ? rev) then shortRev else "${shortRev}-dirty";

        # Image metadata for attestation
        imageMetadata = {
          "org.opencontainers.image.title" = "Neko Agent";
          "org.opencontainers.image.description" = "AI Agent for TEE deployment";
          "org.opencontainers.image.source" = "https://github.com/your-org/neko_agent";
          "org.opencontainers.image.vendor" = "Neko Agent Team";
          "org.opencontainers.image.created" = timestamp;
          "org.opencontainers.image.revision" = revision;
          "org.opencontainers.image.version" = version;
          "dev.nix.flake.revision" = revision;
          "dev.nix.nixpkgs.revision" = nixpkgsRev;
          "dev.neko.build.reproducible" = "true";
          "dev.neko.build.timestamp" = timestamp;
        };
      };

      # Common overlays used throughout
      nekoOverlays = [
        (import ./overlays/znver2-flags.nix)   # provides pkgs.nekoZnver2Env
        (import ./overlays/vmm-cli.nix)        # provides pkgs.vmm-cli
        # ml-pkgs.overlays.torch-family          # ml-pkgs torch overlay first
        (import ./overlays/pylibsrtp.nix)      # python packages after torch
        (import ./overlays/aioice.nix)
        (import ./overlays/aiortc.nix)
        (import ./overlays/streaming.nix)
        (import ./overlays/cached-path.nix)
        (import ./overlays/ema-pytorch.nix)
        (import ./overlays/vocos.nix)
        (import ./overlays/transformers-stream-generator.nix)
        (import ./overlays/bitsandbytes.nix)
        (import ./overlays/f5-tts.nix)
        # pi-zero-pytorch dependencies
        (import ./overlays/pi-zero-pytorch/accelerated-scan.nix)
        (import ./overlays/pi-zero-pytorch/assoc-scan.nix)
        (import ./overlays/pi-zero-pytorch/bidirectional-cross-attention.nix)
        (import ./overlays/pi-zero-pytorch/einx.nix)
        (import ./overlays/pi-zero-pytorch/hl-gauss-pytorch.nix)
        (import ./overlays/pi-zero-pytorch/adam-atan2-pytorch.nix)
        (import ./overlays/pi-zero-pytorch/evolutionary-policy-optimization.nix)
        (import ./overlays/pi-zero-pytorch/hyper-connections.nix)
        (import ./overlays/pi-zero-pytorch/rotary-embedding-torch.nix)
        (import ./overlays/pi-zero-pytorch/torchtyping.nix)
        (import ./overlays/pi-zero-pytorch/x-mlps-pytorch.nix)
        (import ./overlays/pi-zero-pytorch/x-transformers.nix)
        (import ./overlays/einops.nix)         # disable problematic tests - put last to avoid conflicts
        (import ./overlays/pi-zero-pytorch/pi-zero-pytorch.nix)
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

      # Helper function to sort keys deterministically (matching dstack SDK)
      sortKeys = obj:
        if builtins.isAttrs obj then
          builtins.listToAttrs (
            map (key: { name = key; value = sortKeys obj.${key}; })
            (builtins.sort (a: b: a < b) (builtins.attrNames obj))
          )
        else if builtins.isList obj then
          map sortKeys obj
        else
          obj;

      # Dstack app-compose generator for reproducible deployments
      generateAppCompose = pkgs: {
        name,
        dockerComposeContent,
        imageDigest ? null,
        features ? {}
      }: let
        # Deterministic compose configuration
        appCompose = sortKeys {
          manifest_version = 2;
          inherit name;
          runner = "docker-compose";
          docker_compose_file = dockerComposeContent;

          # Security settings
          gateway_enabled = features.gateway or true;
          kms_enabled = features.kms or false;
          public_logs = features.publicLogs or false;
          public_sysinfo = features.publicSysinfo or true;
          public_tcbinfo = features.publicTcbinfo or true;
          local_key_provider_enabled = features.localKeyProvider or false;
          no_instance_id = features.noInstanceId or false;

          # Optional pre-launch script for attestation
          pre_launch_script = features.preLaunchScript or null;
        };

        # Generate JSON with deterministic serialization
        composeJsonContent = builtins.toJSON appCompose;
        composeJson = pkgs.writeText "app-compose.json" composeJsonContent;

        # Calculate compose hash using same algorithm as dstack SDK
        composeHash = builtins.hashString "sha256" composeJsonContent;

      in {
        json = composeJson;
        hash = composeHash;
        content = appCompose;
        jsonContent = composeJsonContent;
      };

      # Container registry configuration with multiple registry support
      registryConfig = rec {
        # Default registry - can be overridden via environment
        defaultRegistry = let env = builtins.getEnv "NEKO_REGISTRY"; in if env != "" then env else "ghcr.io/your-org";

        # Registry-specific settings
        registries = {
          "ghcr.io" = {
            prefix = "ghcr.io/your-org";
            supportsDigests = true;
            requiresAuth = true;
            ttl = null;
          };
          "docker.io" = {
            prefix = "docker.io/your-org";
            supportsDigests = true;
            requiresAuth = true;
            ttl = null;
          };
          "ttl.sh" = {
            prefix = "ttl.sh";
            supportsDigests = false;  # ttl.sh uses time-based tags
            requiresAuth = false;
            ttl = "1h";  # default TTL for ttl.sh
            anonymous = true;
          };
          "localhost:5000" = {
            prefix = "localhost:5000/neko";
            supportsDigests = true;
            requiresAuth = false;
            ttl = null;
          };
        };

        # Helper to get registry config
        getRegistryConfig = registry:
          let
            # Extract base domain from registry string
            baseDomain = if (builtins.match ".*://.*" registry != null)
              then builtins.head (builtins.match "https?://([^/]+).*" registry)
              else if (builtins.match "[^/]+/.*" registry != null)
              then builtins.head (builtins.match "([^/]+)/.*" registry)
              else registry;
          in
          (registries.${baseDomain} or (registries.${registry} or {
            prefix = registry;
            supportsDigests = true;
            requiresAuth = true;
            ttl = null;
          }));

        # Generate image name with registry prefix
        generateImageName = { baseName, registry ? defaultRegistry }:
          let
            config = getRegistryConfig registry;
            # For ttl.sh, use UUID-based naming for anonymity
            imageName = if (config.anonymous or false)
              then "${config.prefix}/$(uuidgen | tr '[:upper:]' '[:lower:]')"
              else "${config.prefix}/${baseName}";
          in imageName;

        # Generate appropriate tag based on registry capabilities
        generateTag = { variant, registry ? defaultRegistry, ttl ? null }:
          let
            config = getRegistryConfig registry;
            # Use TTL tags for ttl.sh, version tags for others
            tag = if (config.ttl or null) != null
              then "${variant}-${buildInfo.shortRev}:${if ttl != null then ttl else config.ttl}"
              else "${variant}-${buildInfo.shortRev}";
          in tag;
      };

      # Enhanced image builder with reproducible settings and registry support
      buildReproducibleImage = pkgs: cuda: {
        name,
        variant,  # "generic" or "opt"
        runner,
        pyEnv,
        extraPackages ? [],
        config ? {},
        registry ? registryConfig.defaultRegistry,
        ttl ? null
      }:
        let
          registryConf = registryConfig.getRegistryConfig registry;
          finalName = if (registryConf.anonymous or false)
            then "${registryConf.prefix}/$(uuidgen | tr '[:upper:]' '[:lower:]')"
            else name;
          finalTag = registryConfig.generateTag { inherit variant registry ttl; };
        in
        pkgs.dockerTools.buildImage {
          name = finalName;
          tag = finalTag;

        # Fixed creation time for reproducibility
        created = buildInfo.timestamp;

        copyToRoot = (pkgs.buildEnv {
          name = "image-root";
          paths = [
            runner
            pyEnv
            cuda.cudatoolkit
            cuda.cudnn
            cuda.nccl
            pkgs.bashInteractive
          ] ++ extraPackages ++ (mkCommonSystemPackages pkgs);
          pathsToLink = [ "/bin" ];
        });

        config = {
          Env = [
            "PYTHONUNBUFFERED=1"
            "NEKO_LOGLEVEL=INFO"
            "CUDA_HOME=${cuda.cudatoolkit}"
            "LD_LIBRARY_PATH=${cuda.cudatoolkit}/lib64:${cuda.cudnn}/lib"
            "CUDA_MODULE_LOADING=LAZY"
            "NEKO_BUILD_VERSION=${buildInfo.version}"
            "NEKO_BUILD_TIMESTAMP=${buildInfo.timestamp}"
          ] ++ (config.Env or []);

          WorkingDir = "/workspace";
          Entrypoint = [ "/bin/${runner.name or "neko-agent"}" ];

          # OCI metadata for attestation
          Labels = buildInfo.imageMetadata // {
            "dev.neko.variant" = variant;
            "dev.neko.binary" = runner.name or "neko-agent";
          } // (config.Labels or {});
        } // (builtins.removeAttrs config ["Env" "Labels"]);
      };

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
            # Note: pylibsrtp, aioice, aiortc, streaming, f5-tts are handled as standalone packages via overlays
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


              pkgs.python3Packages.pi-zero-pytorch
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
              pkgs.python3Packages.pylibsrtp
              pkgs.python3Packages.aioice
              pkgs.python3Packages.aiortc
              pkgs.python3Packages.streaming
              pkgs.python3Packages.f5-tts
              pkgs.python3Packages.pi-zero-pytorch
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
                COMPOSE_FILE="${./docker-compose/neko-server.yaml}"
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
              pkgs.jq
              (pkgs.python3.withPackages (ps: [
                ps.setuptools  # For future dstack SDK if installed
              ]))
            ];
            shellHook = ''
              ${loadDotenv}
              echo "🔐 TEE Development Environment (Attestation-Ready)"
              echo "Build Info: ${buildInfo.version} (${buildInfo.timestamp})"
              echo "Git Revision: ${buildInfo.revision}"
              echo ""
              echo "Tools available:"
              echo "  - Phala Cloud CLI (phala)"
              echo "  - Legacy VMM CLI (vmm-cli)"
              echo "  - Docker & Docker Compose"
              echo "  - Bun runtime"
              echo "  - Reproducible image builder"
              echo "  - Attestation tooling"
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
              echo "🔐 Attestation Commands:"
              echo "  nix run .#build-images             - Build reproducible images"
              echo "  nix run .#deploy-to-tee            - Deploy with attestation metadata"
              echo "  nix run .#verify-attestation       - Verify TEE attestation"
              echo ""
              echo "🐳 Container Registry Commands:"
              echo "  nix run .#push-to-ttl [TTL]        - Push to ttl.sh ephemeral registry"
              echo "  nix run .#deploy-to-ttl [TTL]      - Deploy to TEE using ttl.sh"
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
              echo "🔍 For reproducible deployments:"
              echo "  1. Run 'nix run .#build-images' to build reproducible images"
              echo "  2. Run 'nix run .#deploy-to-tee' to deploy with attestation"
              echo "  3. Use compose hash from deployment for verification"
              echo ""
              echo "🌐 Multi-Registry Examples:"
              echo "  ttl.sh (1h):     NEKO_REGISTRY=ttl.sh NEKO_TTL=1h nix run .#deploy-to-tee"
              echo "  ttl.sh (24h):    nix run .#deploy-to-ttl 24h"
              echo "  GitHub:          NEKO_REGISTRY=ghcr.io/your-org nix run .#deploy-to-tee"
              echo "  Docker Hub:      NEKO_REGISTRY=docker.io/your-org nix run .#deploy-to-tee"
              echo "  Local registry:  NEKO_REGISTRY=localhost:5000 nix run .#deploy-to-tee"
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
            # ps.pylibsrtp
            # ps.aioice
            # ps.aiortc
            # ps.streaming
            # ps.f5-tts
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

          # Additional runners for other scripts
          captureRunnerGeneric = pkgs.writeShellScriptBin "neko-capture" ''
            exec ${pyEnvGeneric}/bin/python ${./src/capture.py} "$@"
          '';
          captureRunnerOpt = pkgs.writeShellScriptBin "neko-capture" ''
            exec ${pyEnvOpt}/bin/python ${./src/capture.py} "$@"
          '';

          yapRunnerGeneric = pkgs.writeShellScriptBin "neko-yap" ''
            exec ${pyEnvGeneric}/bin/python ${./src/yap.py} "$@"
          '';
          yapRunnerOpt = pkgs.writeShellScriptBin "neko-yap" ''
            exec ${pyEnvOpt}/bin/python ${./src/yap.py} "$@"
          '';

          trainRunnerGeneric = pkgs.writeShellScriptBin "neko-train" ''
            exec ${pyEnvGeneric}/bin/python ${./src/train.py} "$@"
          '';
          trainRunnerOpt = pkgs.writeShellScriptBin "neko-train" ''
            exec ${pyEnvOpt}/bin/python ${./src/train.py} "$@"
          '';

          baseEnv = [
            "PYTHONUNBUFFERED=1"
            "NEKO_LOGLEVEL=INFO"
          ];

          # Docker-compose content generator with digest pinning
          dockerComposeWithDigests = imageDigest: ''
            version: '3.8'
            services:
              neko-agent:
                image: ghcr.io/your-org/neko-agent@${imageDigest}
                restart: unless-stopped
                environment:
                  - NEKO_LOGLEVEL=INFO
                  - NEKO_BUILD_VERSION=${buildInfo.version}
                ports:
                  - "8080:8080"
                volumes:
                  - /var/run/dstack.sock:/var/run/dstack.sock
          '';

          # Reproducible image builder instances
          buildImage = buildReproducibleImage pkgs cuda;
        in
        {
          # Reproducible attestation-ready Docker images
          neko-agent-docker-generic = buildImage {
            name = "ghcr.io/your-org/neko-agent";
            variant = "generic";
            runner = runnerGeneric;
            pyEnv = pyEnvGeneric;
            config.Env = [
              "TORCH_CUDA_ARCH_LIST=8.6+PTX"
            ];
          };

          neko-agent-docker-opt = buildImage {
            name = "ghcr.io/your-org/neko-agent";
            variant = "opt";
            runner = runnerOpt;
            pyEnv = pyEnvOpt;
            extraPackages = [
              (pkgs.writeShellScriptBin "neko-znver2-env" "source ${pkgs.nekoZnver2Env}; exec \"$@\"")
            ];
            config = {
              Env = [
                "TORCH_CUDA_ARCH_LIST=8.6"
                "NIX_CFLAGS_COMPILE=-O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
                "RUSTFLAGS=-C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
              ];
            };
          };

          # Capture images
          neko-capture-docker-generic = buildImage {
            name = "ghcr.io/your-org/neko-capture";
            variant = "generic";
            runner = captureRunnerGeneric;
            pyEnv = pyEnvGeneric;
            config.Env = [
              "TORCH_CUDA_ARCH_LIST=8.6+PTX"
            ];
          };

          neko-capture-docker-opt = buildImage {
            name = "ghcr.io/your-org/neko-capture";
            variant = "opt";
            runner = captureRunnerOpt;
            pyEnv = pyEnvOpt;
            extraPackages = [
              (pkgs.writeShellScriptBin "neko-znver2-env" "source ${pkgs.nekoZnver2Env}; exec \"$@\"")
            ];
            config = {
              Env = [
                "TORCH_CUDA_ARCH_LIST=8.6"
                "NIX_CFLAGS_COMPILE=-O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
                "RUSTFLAGS=-C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
              ];
            };
          };

          # Yap images
          neko-yap-docker-generic = buildImage {
            name = "ghcr.io/your-org/neko-yap";
            variant = "generic";
            runner = yapRunnerGeneric;
            pyEnv = pyEnvGeneric;
            config.Env = [
              "TORCH_CUDA_ARCH_LIST=8.6+PTX"
            ];
          };

          neko-yap-docker-opt = buildImage {
            name = "ghcr.io/your-org/neko-yap";
            variant = "opt";
            runner = yapRunnerOpt;
            pyEnv = pyEnvOpt;
            extraPackages = [
              (pkgs.writeShellScriptBin "neko-znver2-env" "source ${pkgs.nekoZnver2Env}; exec \"$@\"")
            ];
            config = {
              Env = [
                "TORCH_CUDA_ARCH_LIST=8.6"
                "NIX_CFLAGS_COMPILE=-O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
                "RUSTFLAGS=-C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
              ];
            };
          };

          # Train images
          neko-train-docker-generic = buildImage {
            name = "ghcr.io/your-org/neko-train";
            variant = "generic";
            runner = trainRunnerGeneric;
            pyEnv = pyEnvGeneric;
            config.Env = [
              "TORCH_CUDA_ARCH_LIST=8.6+PTX"
            ];
          };

          neko-train-docker-opt = buildImage {
            name = "ghcr.io/your-org/neko-train";
            variant = "opt";
            runner = trainRunnerOpt;
            pyEnv = pyEnvOpt;
            extraPackages = [
              (pkgs.writeShellScriptBin "neko-znver2-env" "source ${pkgs.nekoZnver2Env}; exec \"$@\"")
            ];
            config = {
              Env = [
                "TORCH_CUDA_ARCH_LIST=8.6"
                "NIX_CFLAGS_COMPILE=-O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
                "RUSTFLAGS=-C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
              ];
            };
          };

        };

      # helper to build all images
      apps.x86_64-linux.build-images =
        let
          pkgs-app = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            overlays = nekoOverlays;
          };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "build-images" ''
            set -euo pipefail
            echo "🔨 Building reproducible images with attestation metadata..."
            echo "Build version: ${buildInfo.version}"
            echo "Build timestamp: ${buildInfo.timestamp}"
            echo "Git revision: ${buildInfo.revision}"
            echo ""
            echo "Building agent images..."
            nix build .#neko-agent-docker-generic
            nix build .#neko-agent-docker-opt
            echo "Building capture images..."
            nix build .#neko-capture-docker-generic
            nix build .#neko-capture-docker-opt
            echo "Building yap images..."
            nix build .#neko-yap-docker-generic
            nix build .#neko-yap-docker-opt
            echo "Building train images..."
            nix build .#neko-train-docker-generic
            nix build .#neko-train-docker-opt
            echo "✅ All reproducible images built successfully."
          '');
        };

      # TEE deployment tooling with multi-registry support
      apps.x86_64-linux.deploy-to-tee =
        let
          pkgs-app = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            overlays = nekoOverlays;
          };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "deploy-to-tee" ''
            set -euo pipefail

            # Load environment variables if .env exists
            if [ -f .env ]; then
              set -a; source .env; set +a
            fi

            # Registry configuration - can be overridden via environment
            REGISTRY=''${NEKO_REGISTRY:-ghcr.io/your-org}
            TTL=''${NEKO_TTL:-}
            PUSH_IMAGE=''${NEKO_PUSH:-true}

            echo "🔐 Neko TEE Deployment (Attestation-Ready)"
            echo "=========================================="
            echo "Build: ${buildInfo.version} (${buildInfo.timestamp})"
            echo "Registry: $REGISTRY"
            echo "VMM URL: ''${DSTACK_VMM_URL:-not set}"
            echo ""

            # Function to generate UUID for anonymous registries like ttl.sh
            generate_uuid() {
              if command -v uuidgen &> /dev/null; then
                uuidgen | tr '[:upper:]' '[:lower:]'
              else
                # Fallback UUID generation using /dev/urandom
                cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 32 | head -n 1 | sed 's/\(..\)/\1-/g' | sed 's/-$//'
              fi
            }

            # Function to get registry-specific configuration
            get_registry_config() {
              local registry=$1
              case "$registry" in
                ttl.sh*)
                  echo "anonymous=true"
                  echo "ttl=''${TTL:-1h}"
                  echo "auth_required=false"
                  echo "supports_digests=false"
                  ;;
                ghcr.io*)
                  echo "anonymous=false"
                  echo "auth_required=true"
                  echo "supports_digests=true"
                  ;;
                docker.io*)
                  echo "anonymous=false"
                  echo "auth_required=true"
                  echo "supports_digests=true"
                  ;;
                localhost:*)
                  echo "anonymous=false"
                  echo "auth_required=false"
                  echo "supports_digests=true"
                  ;;
                *)
                  echo "anonymous=false"
                  echo "auth_required=true"
                  echo "supports_digests=true"
                  ;;
              esac
            }

            # Get registry configuration
            REGISTRY_CONFIG=$(get_registry_config "$REGISTRY")
            eval "$REGISTRY_CONFIG"

            # Generate image name based on registry type
            if [ "''${anonymous:-false}" = "true" ]; then
              IMAGE_UUID=$(generate_uuid)
              if [ -n "$TTL" ]; then
                IMAGE_NAME="$REGISTRY/$IMAGE_UUID:$TTL"
                DEPLOY_IMAGE_NAME="$REGISTRY/$IMAGE_UUID:$TTL"
              else
                IMAGE_NAME="$REGISTRY/$IMAGE_UUID:${buildInfo.shortRev}"
                DEPLOY_IMAGE_NAME="$REGISTRY/$IMAGE_UUID:${buildInfo.shortRev}"
              fi
              echo "🎲 Anonymous registry detected, using UUID: $IMAGE_UUID"
            else
              IMAGE_NAME="$REGISTRY/neko-agent:opt-${buildInfo.shortRev}"
              DEPLOY_IMAGE_NAME="$IMAGE_NAME"
            fi

            echo "📦 Target image: $IMAGE_NAME"

            # Step 1: Build reproducible images
            echo "📦 Building reproducible images..."
            nix build .#neko-agent-docker-generic
            nix build .#neko-agent-docker-opt

            # Step 2: Load and get actual digests
            echo "🔍 Loading images and extracting digests..."
            if ! command -v docker &> /dev/null; then
                echo "❌ Docker not available. Please ensure Docker is running."
                exit 1
            fi

            GENERIC_IMAGE=$(docker load < result | grep "Loaded image" | cut -d' ' -f3)
            GENERIC_DIGEST=$(docker inspect "$GENERIC_IMAGE" --format='{{index .Id}}')

            echo "✅ Generic image: $GENERIC_IMAGE"
            echo "✅ Image digest: $GENERIC_DIGEST"

            # Step 3: Tag and optionally push to target registry
            if [ "$PUSH_IMAGE" = "true" ]; then
              echo "🏷️  Tagging image for target registry..."
              docker tag "$GENERIC_IMAGE" "$IMAGE_NAME"

              echo "📤 Pushing to registry: $REGISTRY"
              if [ "''${auth_required:-true}" = "true" ] && [ "''${anonymous:-false}" != "true" ]; then
                echo "🔐 Authentication may be required for $REGISTRY"
                echo "   Make sure you're logged in with: docker login $REGISTRY"
              fi

              # Special handling for ttl.sh - show the one-liner approach
              if [[ "$REGISTRY" == ttl.sh* ]]; then
                echo "💡 ttl.sh one-liner approach:"
                echo "   IMAGE=$IMAGE_UUID"
                echo "   docker build -t ttl.sh/\$IMAGE:''${TTL:-1h} ."
                echo "   docker push ttl.sh/\$IMAGE:''${TTL:-1h}"
              fi

              if docker push "$IMAGE_NAME"; then
                echo "✅ Image pushed successfully"
                DEPLOY_IMAGE_NAME="$IMAGE_NAME"
              else
                echo "⚠️  Push failed, using local image"
                DEPLOY_IMAGE_NAME="$GENERIC_IMAGE"
              fi
            else
              echo "🚫 Skipping image push (NEKO_PUSH=false)"
              DEPLOY_IMAGE_NAME="$GENERIC_IMAGE"
            fi

            # Step 4: Generate app-compose with appropriate image reference
            echo "⚙️  Generating app-compose with pinned digests..."

            # Use digest pinning for registries that support it
            if [ "''${supports_digests:-true}" = "true" ] && [ "$PUSH_IMAGE" = "true" ]; then
              # Try to get the pushed image digest
              PUSHED_DIGEST=$(docker inspect "$IMAGE_NAME" --format='{{index .RepoDigests 0}}' 2>/dev/null || echo "")
              if [ -n "$PUSHED_DIGEST" ]; then
                FINAL_IMAGE_REF="$PUSHED_DIGEST"
                echo "🔒 Using digest pinning: $FINAL_IMAGE_REF"
              else
                FINAL_IMAGE_REF="$DEPLOY_IMAGE_NAME"
                echo "📌 Using tag reference: $FINAL_IMAGE_REF"
              fi
            else
              FINAL_IMAGE_REF="$DEPLOY_IMAGE_NAME"
              echo "📌 Using tag reference: $FINAL_IMAGE_REF"
            fi

            COMPOSE_CONTENT=$(cat <<EOF
version: '3.8'
services:
  neko-agent:
    image: $FINAL_IMAGE_REF
    restart: unless-stopped
    environment:
      - NEKO_LOGLEVEL=INFO
      - NEKO_BUILD_VERSION=${buildInfo.version}
      - NEKO_REGISTRY=$REGISTRY
    ports:
      - "8080:8080"
    volumes:
      - /var/run/dstack.sock:/var/run/dstack.sock
EOF
)

            # Generate app-compose.json with proper JSON escaping
            ESCAPED_COMPOSE=$(echo "$COMPOSE_CONTENT" | jq -Rs .)
            cat > neko-app-compose.json <<EOF
{
  "manifest_version": 2,
  "name": "neko-agent-reproducible",
  "runner": "docker-compose",
  "docker_compose_file": $ESCAPED_COMPOSE,
  "gateway_enabled": true,
  "public_logs": false,
  "public_tcbinfo": true
}
EOF

            # Calculate compose hash (simplified version)
            COMPOSE_HASH=$(cat neko-app-compose.json | sha256sum | cut -d' ' -f1)

            # Generate metadata
            cat > compose-metadata.json <<EOF
{
  "compose_hash": "$COMPOSE_HASH",
  "build_version": "${buildInfo.version}",
  "build_timestamp": "${buildInfo.timestamp}",
  "image_digest": "$GENERIC_DIGEST",
  "image_name": "$IMAGE_NAME",
  "image_ref": "$FINAL_IMAGE_REF",
  "registry": "$REGISTRY",
  "nix_revision": "${buildInfo.revision}",
  "reproducible": true
}
EOF

            echo "✅ Generated attestable app-compose.json"
            echo "📋 Metadata:"
            cat compose-metadata.json | ${pkgs-app.jq}/bin/jq .

            # Step 5: Deploy with vmm-cli (using overlay)
            echo "🚀 Deploying to TEE..."

            # Show registry-specific information
            if [[ "$REGISTRY" == ttl.sh* ]]; then
              echo "🕒 Using ttl.sh ephemeral registry"
              echo "   Image will expire after: ''${TTL:-1h}"
              echo "   Pull within the TTL window on another machine:"
              echo "   docker pull $FINAL_IMAGE_REF"
            fi

            APP_ID=$(echo -n "$COMPOSE_HASH" | cut -c1-40)
            echo "App ID: $APP_ID"

            if ${pkgs-app.vmm-cli}/bin/vmm-cli deploy \
              --name "neko-reproducible-''${USER:-unknown}" \
              --image "dstack-nvidia-0.5.3" \
              --compose ./neko-app-compose.json \
              --vcpu 4 \
              --memory 8G \
              --disk 50G; then
                echo "🎉 Deployment complete!"
            else
                echo "⚠️  vmm-cli deployment failed. Generated files for manual deployment:"
                echo "   - neko-app-compose.json"
                echo "   - compose-metadata.json"
            fi

            echo ""
            echo "🔍 For attestation verification:"
            echo "   Compose hash: $COMPOSE_HASH"
            echo "   This hash will be recorded in RTMR3 for verification"
            echo ""
            echo "🎯 Registry Usage Examples:"
            echo "   ttl.sh (1h):     NEKO_REGISTRY=ttl.sh NEKO_TTL=1h nix run .#deploy-to-tee"
            echo "   ttl.sh (24h):    NEKO_REGISTRY=ttl.sh NEKO_TTL=24h nix run .#deploy-to-tee"
            echo "   GitHub:          NEKO_REGISTRY=ghcr.io/your-org nix run .#deploy-to-tee"
            echo "   Docker Hub:      NEKO_REGISTRY=docker.io/your-org nix run .#deploy-to-tee"
            echo "   Local registry:  NEKO_REGISTRY=localhost:5000/neko nix run .#deploy-to-tee"
          '');
        };

      # Quick ttl.sh deployment app
      apps.x86_64-linux.deploy-to-ttl =
        let
          pkgs-app = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            overlays = nekoOverlays;
          };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "deploy-to-ttl" ''
            set -euo pipefail

            TTL=''${1:-1h}
            echo "🕒 Quick ttl.sh deployment with TTL: $TTL"
            echo "====================================="
            echo ""

            # Set environment for ttl.sh
            export NEKO_REGISTRY=ttl.sh
            export NEKO_TTL=$TTL

            echo "Using ttl.sh ephemeral registry with $TTL TTL"
            echo "Image will be anonymous and expire automatically"
            echo ""

            # Call the main deploy-to-tee script
            exec ${toString self.apps.x86_64-linux.deploy-to-tee.program}
          '');
        };

      # Build and push to ttl.sh (without TEE deployment)
      apps.x86_64-linux.push-to-ttl =
        let
          pkgs-app = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            overlays = nekoOverlays;
          };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "push-to-ttl" ''
            set -euo pipefail

            TTL=''${1:-1h}
            echo "📤 Building and pushing to ttl.sh (TTL: $TTL)"
            echo "============================================="
            echo ""

            # Function to generate UUID for anonymous registries like ttl.sh
            generate_uuid() {
              if command -v uuidgen &> /dev/null; then
                uuidgen | tr '[:upper:]' '[:lower:]'
              else
                # Fallback UUID generation using /dev/urandom
                cat /dev/urandom | tr -dc 'a-f0-9' | fold -w 32 | head -n 1 | sed 's/\(..\)/\1-/g' | sed 's/-$//'
              fi
            }

            IMAGE_UUID=$(generate_uuid)
            IMAGE_NAME="ttl.sh/$IMAGE_UUID:$TTL"

            echo "🎲 Generated UUID: $IMAGE_UUID"
            echo "📦 Target image: $IMAGE_NAME"
            echo ""

            # Build the image
            echo "🔨 Building reproducible image..."
            nix build .#neko-agent-docker-opt

            if ! command -v docker &> /dev/null; then
                echo "❌ Docker not available. Please ensure Docker is running."
                exit 1
            fi

            # Load and tag
            echo "🔍 Loading image..."
            BUILT_IMAGE=$(docker load < result | grep "Loaded image" | cut -d' ' -f3)

            echo "🏷️  Tagging for ttl.sh..."
            docker tag "$BUILT_IMAGE" "$IMAGE_NAME"

            echo "📤 Pushing to ttl.sh..."
            echo "💡 ttl.sh is anonymous - no authentication required"

            if docker push "$IMAGE_NAME"; then
              echo "✅ Image pushed successfully to ttl.sh!"
              echo ""
              echo "🔗 Your image: $IMAGE_NAME"
              echo "⏰ Expires after: $TTL"
              echo ""
              echo "📋 Pull on another machine within $TTL:"
              echo "   docker pull $IMAGE_NAME"
              echo "   docker run --rm -p 8080:8080 $IMAGE_NAME"
              echo ""
              echo "🎯 For TEE deployment:"
              echo "   NEKO_REGISTRY=ttl.sh NEKO_TTL=$TTL nix run .#deploy-to-tee"
            else
              echo "❌ Push failed"
              exit 1
            fi
          '');
        };

      # Attestation verification tool
      apps.x86_64-linux.verify-attestation =
        let
          pkgs-app = import nixpkgs {
            system = "x86_64-linux";
            config.allowUnfree = true;
            overlays = nekoOverlays;
          };
        in
        {
          type = "app";
          program = toString (pkgs-app.writeShellScript "verify-attestation" ''
            set -euo pipefail

            if [ $# -lt 2 ]; then
                echo "Usage: verify-attestation <app-id> <expected-hash>"
                echo ""
                echo "Example:"
                echo "  verify-attestation abc123def456 1234abcd..."
                exit 1
            fi

            APP_ID=$1
            EXPECTED_HASH=$2

            echo "🔍 Verifying TEE attestation for app: $APP_ID"
            echo "Expected compose hash: $EXPECTED_HASH"
            echo ""

            # Check if we're in a TEE environment
            if [ ! -S /var/run/dstack.sock ]; then
                echo "❌ Not running in a TEE environment"
                echo "   /var/run/dstack.sock not found"
                echo ""
                echo "Manual verification steps:"
                echo "1. Deploy the app with reproducible images"
                echo "2. Get TDX quote from within the TEE"
                echo "3. Verify quote signature with dcap-qvl"
                echo "4. Check RTMR3 contains expected compose hash"
                echo "5. Verify other measurements match expected image"
                exit 1
            fi

            # Get TDX quote with compose hash as report data
            echo "📋 Requesting TDX quote with compose hash as report data..."
            if ! ${pkgs-app.curl}/bin/curl --unix-socket /var/run/dstack.sock \
                "http://localhost/GetQuote?report_data=0x$EXPECTED_HASH" > quote_response.json; then
                echo "❌ Failed to get TDX quote"
                exit 1
            fi

            # Extract and save quote
            ${pkgs-app.jq}/bin/jq -r '.quote' quote_response.json | base64 -d > quote.bin

            echo "✅ Quote generated successfully (''${#EXPECTED_HASH} bytes)"
            echo "📄 Quote saved to: quote.bin"
            echo ""
            echo "🔍 Manual verification steps:"
            echo "1. Verify quote signature with Intel DCAP libraries"
            echo "2. Extract and verify RTMR3 contains: $EXPECTED_HASH"
            echo "3. Verify MRTD, RTMR0, RTMR1, RTMR2 match expected image measurements"
            echo ""
            echo "✅ Quote generation complete - manual verification required"
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
