self: super: {
  python3Packages = super.python3Packages // {
    aioice = super.python3Packages.buildPythonPackage rec {
      pname    = "aioice";
      version  = "0.8.0";
      format   = "setuptools";
      src      = super.fetchFromGitHub {
        owner = "aiortc"; repo = "aioice"; rev = "v${version}";
        sha256 = "sha256-KFYPzGPm+d1QrFAW9OhTDxroV/MnFusmfy5qcYCfDiM=";
      };
      propagatedBuildInputs = with super.python3Packages; [ dnspython ifaddr ];
      doCheck = false;
      meta = with super.lib; {
        description = "Interactive Connectivity Establishment (RFC 5245) in Python";
        homepage    = "https://github.com/aiortc/aioice";
        license     = licenses.bsd3;
      };
    };
  };
}
