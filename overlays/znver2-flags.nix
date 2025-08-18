# overlays/znver2-flags.nix
final: prev: {
  # Tiny script you can `source` to add tuned flags to *builds done inside shells/containers*.
  nekoZnver2Env = prev.writeText "neko-znver2-flags.sh" ''
    # AMD Zen 2 tuning (Threadripper 3960X etc.)
    export NIX_CFLAGS_COMPILE="''${NIX_CFLAGS_COMPILE:+$NIX_CFLAGS_COMPILE } -O3 -pipe -march=znver2 -mtune=znver2 -fno-plt"
    export NIX_LDFLAGS="''${NIX_LDFLAGS:+$NIX_LDFLAGS } -Wl,-O1 -Wl,--as-needed -Wl,--hash-style=gnu"
    export RUSTFLAGS="''${RUSTFLAGS:+$RUSTFLAGS } -C target-cpu=znver2 -C target-feature=+sse2,+sse4.2,+avx,+avx2,+fma,+bmi1,+bmi2 -C link-arg=-Wl,-O1 -C link-arg=--as-needed"
    export OPENBLAS_NUM_THREADS="''${OPENBLAS_NUM_THREADS:-48}"
  '';
}
