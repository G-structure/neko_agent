final: prev: {
  vmm-cli = prev.writeShellScriptBin "vmm-cli" ''
    export PYTHONPATH="${prev.python3.pkgs.makePythonPath (with prev.python3.pkgs; [ cryptography eth-keys eth-utils ])}"
    exec ${prev.python3}/bin/python3 ${prev.fetchurl {
      url = "https://raw.githubusercontent.com/Dstack-TEE/dstack/master/vmm/src/vmm-cli.py";
      sha256 = "15fkvfdnb7mbiqpyjbpy1gv4cnbvy22z7qhb64mqss711pdi0lcw";
    }} "$@"
  '';
}