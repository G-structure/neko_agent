self: super: {
  python3Packages = super.python3Packages // {
    ema-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "ema-pytorch";
      version = "0.7.7";
      format = "setuptools";

      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "ema-pytorch";
        rev = "main";
        sha256 = "sha256-hyDLDlwHKLzJ8l+OYi/Lz/k3+TGV163fS8KlPC60ckM=";
      };

      propagatedBuildInputs = with super.python3Packages; [
        # Use torch from environment, not from nixpkgs
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "A simple way to keep track of an Exponential Moving Average (EMA) version of your Pytorch model";
        homepage = "https://github.com/lucidrains/ema-pytorch";
        license = licenses.mit;
      };
    };
  };
}