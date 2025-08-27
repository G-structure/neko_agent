self: super: {
  python3Packages = super.python3Packages // {
    accelerated-scan = super.python3Packages.buildPythonPackage rec {
      pname = "accelerated-scan";
      version = "0.1.2";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "proger";
        repo = "accelerated-scan";
        rev = "main";
        sha256 = "1yiwxa4vzhym1v90wapfynnl4jzkr5hc25a3rnkrvb8cl3612ckd";
      };

      build-system = with super.python3Packages; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        numpy
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Accelerated Scan operations for JAX and PyTorch";
        homepage = "https://github.com/proger/accelerated-scan";
        license = licenses.mit;
      };
    };
  };
}