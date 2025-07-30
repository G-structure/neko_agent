{ pkgs ? import <nixpkgs> {
    overlays = [
      (import ./overlays/pylibsrtp.nix)  # ← first!
      (import ./overlays/aioice.nix)     # ← second
      (import ./overlays/aiortc.nix)     # ← third
    ];
  }
}:

pkgs.mkShell {
  buildInputs = [
    pkgs.ffmpeg
    pkgs.pkgconf
    pkgs.libvpx

    pkgs.python3Packages.pylibsrtp
    pkgs.python3Packages.aioice
    pkgs.python3Packages.aiortc

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
  ];

  shellHook = ''
    export NEKO_LOGLEVEL=DEBUG
    export NEKO_USER=agent
    export NEKO_PASS=admin
    export NEKO_URL=""
    # export OFFLOAD_FOLDER=""
    # export FRAME_SAVE_PATH=""
    # export CLICK_SAVE_PATH=""
    # export NEKO_ICESERVERS='[{"urls":["stun:stun.l.google.com:19302"]}]'
    # export PYTHONPATH=$PWD/src
  '';
}
