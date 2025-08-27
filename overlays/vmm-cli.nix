final: prev: {
  vmm-cli = prev.writeShellScriptBin "vmm-cli" ''
    # Load environment variables if .env exists
    if [ -f .env ]; then
      set -a; source .env; set +a
    fi
    
    # Set default VMM URL if not provided
    export DSTACK_VMM_URL="''${DSTACK_VMM_URL:-https://web.h200-dstack-01.phala.network:40081/}"
    
    # Set Python path for dependencies
    export PYTHONPATH="${prev.python3.pkgs.makePythonPath (with prev.python3.pkgs; [ cryptography eth-keys eth-utils ])}"
    
    # Pass authentication via command line if available
    auth_args=""
    if [ -n "''${DSTACK_VMM_AUTH_USER:-}" ] && [ -n "''${DSTACK_VMM_AUTH_PASSWORD:-}" ]; then
      auth_args="--auth-user=$DSTACK_VMM_AUTH_USER --auth-password=$DSTACK_VMM_AUTH_PASSWORD"
    fi
    
    exec ${prev.python3}/bin/python3 ${prev.fetchurl {
      url = "https://raw.githubusercontent.com/Dstack-TEE/dstack/master/vmm/src/vmm-cli.py";
      sha256 = "15fkvfdnb7mbiqpyjbpy1gv4cnbvy22z7qhb64mqss711pdi0lcw";
    }} --url="$DSTACK_VMM_URL" $auth_args "$@"
  '';
}