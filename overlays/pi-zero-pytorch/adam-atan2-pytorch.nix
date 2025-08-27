self: super: {
  python3Packages = super.python3Packages // {
    adam-atan2-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "adam-atan2-pytorch";
      version = "unstable-2024-08-27";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "adam-atan2-pytorch";
        rev = "main";
        sha256 = "sha256-f7TpYJRoFhoqYMI9mnHOwb7ZWYmEYVQX03RLepoF5e0=";
      };

      build-system = with super.python3Packages; [
        hatchling
      ];

      propagatedBuildInputs = [
        (super.python3Packages."torch-bin")
      ];

      doCheck = false;

      meta = with super.lib; {
        description = "Adam optimizer variant using atan2, in PyTorch";
        homepage = "https://github.com/lucidrains/adam-atan2-pytorch";
        license = licenses.mit;
      };
    };
  };
}
