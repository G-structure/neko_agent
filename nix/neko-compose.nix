{ pkgs, lib, ... }:
let
  composeFile = ../docker-compose.yml;
in
{
  # Helper script and packages for managing the neko compose stack
  environment.systemPackages = with pkgs; [
    docker-compose
    colima
    docker
    (writeShellScriptBin "neko-services" ''
      COMPOSE_FILE="${composeFile}"
      
      # Ensure Colima is running
      ensure_colima() {
        if ! ${colima}/bin/colima status >/dev/null 2>&1; then
          echo "Starting Colima..."
          ${colima}/bin/colima start --vm-type vz --cpu 2 --memory 4 --mount-type sshfs --mount "~:w"
          sleep 5
        fi
        export DOCKER_HOST="unix://$(readlink -f ~/.colima/default/docker.sock)"
      }
      
      case "$1" in
        up)
          echo "Starting Neko browser service..."
          ensure_colima
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d
          echo "Neko browser available at http://localhost:8080"
          echo "Default credentials: neko / admin"
          ;;
        down)
          echo "Stopping Neko browser service..."
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" down
          ;;
        restart)
          echo "Restarting Neko browser service..."
          ensure_colima
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" restart
          ;;
        pull)
          echo "Pulling latest Neko image..."
          ensure_colima
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull
          ;;
        logs)
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" logs -f ''${2:-}
          ;;
        ps)
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps
          ;;
        status)
          echo "=== Colima Status ==="
          if ${colima}/bin/colima status >/dev/null 2>&1; then
            ${colima}/bin/colima status
          else
            echo "Colima not running"
          fi
          echo ""
          echo "=== Neko Service ==="
          ensure_colima
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" ps
          ;;
        update)
          echo "Updating Neko service (pull + restart)..."
          ensure_colima
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" pull
          ${docker-compose}/bin/docker-compose -f "$COMPOSE_FILE" up -d
          ;;
        run-docker)
          echo "Running Neko with direct docker command..."
          ensure_colima
          # Extract the docker run command from compose file
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

  # Set Docker host environment variable for Colima
  environment.variables = {
    DOCKER_HOST = "unix://${builtins.getEnv "HOME"}/.colima/default/docker.sock";
  };
}