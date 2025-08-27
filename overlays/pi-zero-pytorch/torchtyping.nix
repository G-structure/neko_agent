self: super: {
  python3Packages = super.python3Packages // {
    torchtyping = super.python3Packages.buildPythonPackage rec {
      pname = "torchtyping";
      version = "0.1.5";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "patrick-kidger";
        repo = "torchtyping";
        rev = "master";
        sha256 = "194c170chz59j588sxdn7pxgbi1mppvrdlkrl9m3xp5sp1drfvlh";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        typeguard
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Type annotations and dynamic checking for a tensor's shape, dtype, names, etc.";
        homepage = "https://github.com/patrick-kidger/torchtyping";
        license = licenses.asl20;
      };
    };
  };
}
