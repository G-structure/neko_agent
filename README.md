# Getting Started
set these envs in the `shell.nix`
```shell.nix 33-35
    export NEKO_USER=
    export NEKO_PASS=
    export NEKO_URL="https://"
```

```bash
nix-shell
OFFLOAD_FOLDER=""  NEKO_LOGLEVEL=DEBUG python src/agent.py --username --password --neko-url https://
```

Run neko in (1280, 800)
