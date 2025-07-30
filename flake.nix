{
  description = "A development environment for the Neko Agent project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];

      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs {
            inherit system;
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
              export NEKO_URL="https://1debc96c651ca3992733d6b67631d0ec749b557f-52000.dstack-prod5.phala.network"
              export OFFLOAD_FOLDER="/Users/luc/.cache/huggingface"
              export FRAME_SAVE_PATH="/Users/luc/projects/neko_agent/tmp/frame.png"
              export CLICK_SAVE_PATH="/Users/luc/projects/neko_agent/tmp/actions"
              # export NEKO_ICESERVERS='[{"urls":["stun:stun.l.google.com:19302"]}]'
              # export PYTHONPATH=$PWD/src
            '';
          };
        }
      );
    };
}
