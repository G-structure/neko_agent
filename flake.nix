{
  description = "A development environment for the Neko Agent project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Targeted systems for cross-platform support
      supportedSystems = [ "x86_64-linux" "aarch64-darwin" ];

      # Generate a devShell for each system
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems (system:
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
              # Media and system tools
              pkgs.ffmpeg
              pkgs.pkgconf
              pkgs.libvpx

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
            ];

            # Load secret env vars at runtime from an untracked .env
            shellHook = ''
              if [ -f .env ]; then
                # export all variables defined below
                set -a
                # source the file from the project directory
                source .env
                # stop auto-export
                set +a
              fi
            '';
          };
        }
      );
    in
    {
      devShells = forAllSystems;
    };
}
