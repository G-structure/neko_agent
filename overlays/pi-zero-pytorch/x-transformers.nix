self: super: {
  python3Packages = super.python3Packages // {
    x-transformers = super.python3Packages.buildPythonPackage rec {
      pname = "x-transformers";
      version = "2.4.1";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "x-transformers";
        rev = "bdb7b4ea830a77f960d87ff5374c55b91ac81cfc";
        sha256 = "06zmw0y4q8pcgydbs514sjd5x64g8wlg98d6hzwi6lhc1i5ha6nj";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = [
        self.python3Packages.einops
        super.python3Packages.packaging
        (super.python3Packages."torch-bin")
        super.python3Packages.einx
        super.python313Packages.loguru
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "A concise but complete full-attention transformer with a set of promising experimental features from various papers";
        homepage = "https://github.com/lucidrains/x-transformers";
        license = licenses.mit;
      };
    };
  };
}
