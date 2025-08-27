self: super: {
  python3Packages = super.python3Packages // {
    assoc-scan = super.python3Packages.buildPythonPackage rec {
      pname = "assoc-scan";
      version = "0.0.2";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "assoc-scan";
        rev = "main";
        sha256 = "08sx4d0yh6cycd39lbzpiziw3dx532yj7xkkma3x310l0zji9h73";
      };

      build-system = with super.python3Packages; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        accelerated-scan
        einops
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Associative scan operations";
        homepage = "https://github.com/lucidrains/assoc-scan";
        license = licenses.mit;
      };
    };
  };
}