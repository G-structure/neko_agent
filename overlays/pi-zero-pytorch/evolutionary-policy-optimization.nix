self: super: {
  python3Packages = super.python3Packages // {
    evolutionary-policy-optimization = super.python3Packages.buildPythonPackage rec {
      pname = "evolutionary-policy-optimization";
      version = "0.1.19";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "evolutionary-policy-optimization";
        rev = "main";
        sha256 = "1jffvcdrmfs8bzpwv8cfck39gkzqp24yp1nn1ckzzql1w2ngr48y";
      };

      # Uses Hatchling backend per pyproject.toml
      build-system = with super.python3Packages; [
        hatchling
      ];

      propagatedBuildInputs = [
        (super.python3Packages.accelerate.override { torch = super.python3Packages."torch-bin"; })
        (super.python3Packages."adam-atan2-pytorch")
        (super.python3Packages."assoc-scan")
        self.python3Packages.einops
        super.python3Packages.einx
        super.python3Packages."ema-pytorch"
        (super.python3Packages."hl-gauss-pytorch")
        super.python3Packages.tqdm
        (super.python3Packages."torch-bin")
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Pytorch implementation of Evolutionary Policy Optimization";
        homepage = "https://github.com/lucidrains/evolutionary-policy-optimization";
        license = licenses.mit;
      };
    };
  };
}
