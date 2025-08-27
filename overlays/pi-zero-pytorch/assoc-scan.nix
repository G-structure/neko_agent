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
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = [
        (self.python3Packages."accelerated-scan")
        self.python3Packages.einops
        (super.python3Packages."torch-bin")
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
