self: super: {
  python3Packages = super.python3Packages // {
    hl-gauss-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "hl-gauss-pytorch";
      version = "0.1.21";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "hl-gauss-pytorch";
        rev = "main";
        sha256 = "1qslc9qflmnb1m9p1yb38iiqn8inw0wdvhbc1hmf5kchgnj6d5pq";
      };

      build-system = with super.python3Packages; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "The Gaussian Histogram Loss (HL-Gauss) proposed by Imani et al. with a few convenient wrappers for regression, in Pytorch";
        homepage = "https://github.com/lucidrains/hl-gauss-pytorch";
        license = licenses.mit;
      };
    };
  };
}