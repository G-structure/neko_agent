self: super: {
  python3Packages = super.python3Packages // {
    einx = super.python3Packages.buildPythonPackage rec {
      pname = "einx";
      version = "0.3.0";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "fferflo";
        repo = "einx";
        rev = "master";
        sha256 = "0ax9drgi0j7gl40vvahy9i9bmz7d23i0qh14llb7abf06852a39f";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        numpy
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Universal Tensor Operations in Einstein-Inspired Notation for Python";
        homepage = "https://github.com/fferflo/einx";
        license = licenses.mit;
      };
    };
  };
}
