self: super: {
  python3Packages = super.python3Packages // {
    x-mlps-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "x-mlps-pytorch";
      version = "0.0.24";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "x-mlps-pytorch";
        rev = "main";
        sha256 = "1y7kq38y0wfma0rpm20611any4wqx9imxcxymd177sa6s150s1n2";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = [
        self.python3Packages.einops
        (super.python3Packages."torch-bin")
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Just a repository that will house some MLPs and their variants, so to avoid having to reimplement them again and again for different projects";
        homepage = "https://github.com/lucidrains/x-mlps-pytorch";
        license = licenses.mit;
      };
    };
  };
}
