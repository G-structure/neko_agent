self: super: {
  python3Packages = super.python3Packages // {
    pi-zero-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "pi-zero-pytorch";
      version = "0.2.5";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "pi-zero-pytorch";
        rev = "1c13cbbcc236c3cb4fd18213c543188fdf083b33";
        sha256 = "1xl38c5spv798xydljahq8fw9nglwcw95s95zpn11cm9ra4rg4ib";
      };

      nativeBuildInputs = with super.python3Packages; [
        hatchling
      ];

      propagatedBuildInputs = with super.python3Packages; [
        # Core dependencies from pyproject.toml
        (accelerate.override { torch = super.python3Packages."torch-bin"; })
        accelerated-scan
        assoc-scan
        beartype
        bidirectional-cross-attention
        einx
        (self.python3Packages.einops)
        ema-pytorch
        evolutionary-policy-optimization
        hl-gauss-pytorch
        hyper-connections
        jaxtyping
        rotary-embedding-torch
        scipy
        (torchdiffeq.override { torch = super.python3Packages."torch-bin"; })
        torchtyping
        tqdm
        x-mlps-pytorch
        x-transformers
        (super.python3Packages."torch-bin")
        # Note: torch and other ML deps should come from environment/ml-pkgs
      ];

      doCheck = false;  # Skip tests to avoid potential issues with dependencies

      meta = with super.lib; {
        description = "π0 in Pytorch";
        homepage = "https://github.com/lucidrains/pi-zero-pytorch";
        license = licenses.mit;
      };
    };
  };
}
