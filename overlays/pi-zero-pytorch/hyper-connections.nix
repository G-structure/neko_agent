self: super: {
  python3Packages = super.python3Packages // {
    hyper-connections = super.python3Packages.buildPythonPackage rec {
      pname = "hyper-connections";
      version = "0.0.10";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "hyper-connections";
        rev = "main";
        sha256 = "0a0296rdqmwn85qh344j841g6di2x1w8b7m7qi1ijr7b3k5j5lzm";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = [
        (super.python3Packages."torch-bin")
        self.python3Packages.einops
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Attempt to make multiple residual streams from Bytedance's Hyper-Connections paper accessible to the public";
        homepage = "https://github.com/lucidrains/hyper-connections";
        license = licenses.mit;
      };
    };
  };
}
