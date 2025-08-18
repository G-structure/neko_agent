self: super: {
  python3Packages = super.python3Packages // {
    vocos = super.python3Packages.buildPythonPackage rec {
      pname = "vocos";
      version = "0.1.0";
      format = "setuptools";

      src = super.fetchFromGitHub {
        owner = "gemelo-ai";
        repo = "vocos";
        rev = "main";
        sha256 = "sha256-+vMdS/GMNnjlv+CpzRlsZZeayG38GMyH6DcSQRKNOIU=";
      };

      propagatedBuildInputs = with super.python3Packages; [
        (encodec.override { 
          torch = super.python3Packages."torch-bin";
          torchaudio = super.python3Packages."torchaudio-bin";
        })
        librosa
        numpy
        omegaconf
        # Use torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "Vocos: Closing the gap between time-domain and Fourier-based neural vocoders for high-quality audio synthesis";
        homepage = "https://github.com/gemelo-ai/vocos";
        license = licenses.mit;
      };
    };
  };
}