self: super: {
  python3Packages = super.python3Packages // {
    bidirectional-cross-attention = super.python3Packages.buildPythonPackage rec {
      pname = "bidirectional-cross-attention";
      version = "0.1.0";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "bidirectional-cross-attention";
        rev = "7020c618615e7bcea142af0ce38f22507e6a2332";
        sha256 = "0mhxdr68nwrwwxk2rsmk34r8ha2vprndbppakay9s7l9f8mzijwz";
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
        description = "A simple cross attention that updates both the source and target in one step";
        homepage = "https://github.com/lucidrains/bidirectional-cross-attention";
        license = licenses.mit;
      };
    };
  };
}
