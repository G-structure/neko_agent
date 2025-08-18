self: super: {
  python3Packages = super.python3Packages // {
    aiortc = super.python3Packages.buildPythonPackage rec {
      pname    = "aiortc";
      version  = "1.13.0";
      format   = "setuptools";
      src      = super.fetchFromGitHub {
      owner = "aiortc"; repo = "aiortc"; rev = "${version}";
      sha256 = "sha256-f3Ag1YrIkDCO2Cg047Lf+1x18+bzhUAizkh/0RwASMU=";
      };
      propagatedBuildInputs = [
        super.python3Packages.pylibsrtp
        super.python3Packages.aioice
        super.python3Packages.av
        super.python3Packages.cryptography
        super.python3Packages.google-crc32c
        super.python3Packages.pyee
        super.python3Packages.pyopenssl
      ];
      doCheck = false;
      meta = with super.lib; {
        description = "Python WebRTC/ORTC implementation";
        homepage    = "https://github.com/aiortc/aiortc";
        license     = licenses.bsd3;
      };
    };
  };
}
