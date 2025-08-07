# flake.nix
{
  description = "A development environment for the Neko Agent project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];
      devShellsBySystem = nixpkgs.lib.genAttrs supportedSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [
              (import ./overlays/pylibsrtp.nix)
              (import ./overlays/aioice.nix)
              (import ./overlays/aiortc.nix)
            ];
          };
        in
        {
          default = pkgs.mkShell {
            buildInputs = [
              # Media and system tools
              pkgs.ffmpeg
              pkgs.pkgconf
              pkgs.libvpx
              pkgs.just

              # Python SRTP/ICE/RTCP libraries
              pkgs.python3Packages.pylibsrtp
              pkgs.python3Packages.aioice
              pkgs.python3Packages.aiortc

              # Python AI and utility packages
              (pkgs.python3.withPackages (ps: with ps; [
                transformers
                torch
                torchvision
                pillow
                accelerate
                websockets
                prometheus-client
                av
              ]))

              # Node.js runtime and Claude Code CLI
              pkgs.nodejs_20
              pkgs.claude-code
            ];

            shellHook = ''
              if [ -f .env ]; then
                set -a; source .env; set +a
              fi
            '';
          };

          neko = pkgs.mkShell {
            buildInputs = [
              # Docker and container tools
              pkgs.colima
              pkgs.docker
              pkgs.docker-buildx
              pkgs.docker-compose
              
              # Utilities
              pkgs.curl
              pkgs.jq
              
              # Neko services management script
              (pkgs.writeShellScriptBin "neko-services" ''
                COMPOSE_FILE="${./docker-compose.yml}"
                
                # Ensure Colima is running
                ensure_colima() {
                  if ! ${pkgs.colima}/bin/colima status >/dev/null 2>&1; then
                    echo "Starting Colima..."
                    ${pkgs.colima}/bin/colima start --vm-type vz --cpu 2 --memory 4 --mount-type sshfs --mount "~:w"
                    sleep 5
                  fi
                  export DOCKER_HOST="unix://$(readlink -f ~/.colima/default/docker.sock)"
                }
                
                case "$1" in
                  up)
                    echo "Starting Neko browser service..."
                    ensure_colima
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d
                    echo "Neko browser available at http://localhost:8080"
                    echo "Default credentials: neko / admin"
                    ;;
                  down)
                    echo "Stopping Neko browser service..."
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" down
                    ;;
                  restart)
                    echo "Restarting Neko browser service..."
                    ensure_colima
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" restart
                    ;;
                  pull)
                    echo "Pulling latest Neko image..."
                    ensure_colima
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull
                    ;;
                  logs)
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" logs -f ''${2:-}
                    ;;
                  ps)
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps
                    ;;
                  status)
                    echo "=== Colima Status ==="
                    if ${pkgs.colima}/bin/colima status >/dev/null 2>&1; then
                      ${pkgs.colima}/bin/colima status
                    else
                      echo "Colima not running"
                    fi
                    echo ""
                    echo "=== Neko Service ==="
                    ensure_colima
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps
                    ;;
                  update)
                    echo "Updating Neko service (pull + restart)..."
                    ensure_colima
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull
                    ${pkgs.docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d
                    ;;
                  run-docker)
                    echo "Running Neko with direct docker command..."
                    ensure_colima
                    bash "$COMPOSE_FILE"
                    ;;
                  *)
                    echo "Usage: neko-services {up|down|restart|pull|logs|ps|status|update|run-docker}"
                    echo ""
                    echo "Commands:"
                    echo "  up         - Start Neko browser service"
                    echo "  down       - Stop Neko browser service"
                    echo "  restart    - Restart Neko browser service"
                    echo "  pull       - Pull latest Neko image"
                    echo "  logs       - Follow logs"
                    echo "  ps         - Show running containers"
                    echo "  status     - Show Colima and service status"
                    echo "  update     - Pull latest image and restart service"
                    echo "  run-docker - Run using direct docker command"
                    echo ""
                    echo "Neko browser will be available at: http://localhost:8080"
                    echo "Default credentials: neko / admin (or set NEKO_USER/NEKO_PASS)"
                    exit 1
                    ;;
                esac
              '')
            ];

            shellHook = ''
              echo "Neko Docker environment loaded!"
              echo "Available commands:"
              echo "  neko-services up    - Start Neko browser"
              echo "  neko-services down  - Stop Neko browser"
              echo "  neko-services logs  - View logs"
              echo "  neko-services status - Check status"
              echo ""
              echo "Neko will be available at: http://localhost:8080"
              
              # Set Docker host for Colima
              export DOCKER_HOST="unix://$HOME/.colima/default/docker.sock"
            '';
          };
        }
      );
    in
    {
      devShells = devShellsBySystem;
    };
}
