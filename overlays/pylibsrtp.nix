self: super: {
  python3Packages = super.python3Packages // {
    pylibsrtp = super.python3Packages.buildPythonPackage rec {
      pname    = "pylibsrtp";
      version  = "0.8.0";
      format   = "setuptools";

      src = super.fetchFromGitHub {
        owner = "aiortc";
        repo  = "pylibsrtp";
        rev   = "4b727790393f234c74d45413b267cdcda0f2a2c0";
        sha256 = "2y//OD7wc0z3YPhUxv6w/5d7Kisk3PrbY90313DmPYs=";
      };

      nativeBuildInputs = [
        super.pkgconf
      ];
      buildInputs = [
        super.srtp
        super.openssl
      ];

      propagatedBuildInputs = with super.python3Packages; [ cffi ];

      doCheck = false;

      meta = with super.lib; {
        description = "Python wrapper around the libsrtp library";
        homepage    = "https://github.com/aiortc/pylibsrtp";
        license     = licenses.bsd3;
      };
    };
  };
}
