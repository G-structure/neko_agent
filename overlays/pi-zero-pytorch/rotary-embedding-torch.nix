self: super: {
  python3Packages = super.python3Packages // {
    rotary-embedding-torch = super.python3Packages.buildPythonPackage rec {
      pname = "rotary-embedding-torch";
      version = "0.8.5";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "rotary-embedding-torch";
        rev = "0e44baf61e9999f300ea9db8756ba22d4077ca2e";
        sha256 = "0wvnnk1jy5m2yib3dxss688j5bfjpklk463a53vpxdli96s8xy4q";
      };

      build-system = with super.python3Packages; [
        hatchling
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        einops
        # torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Implementation of Rotary Embeddings, from the Roformer paper, in Pytorch";
        homepage = "https://github.com/lucidrains/rotary-embedding-torch";
        license = licenses.mit;
      };
    };
  };
}
