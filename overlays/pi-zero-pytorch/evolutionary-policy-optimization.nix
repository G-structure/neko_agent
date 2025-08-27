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

      build-system = with super.python3Packages; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        # Add dependencies when known
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