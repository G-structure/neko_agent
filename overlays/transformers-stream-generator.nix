self: super: {
  python3Packages = super.python3Packages // {
    transformers-stream-generator = super.python3Packages.buildPythonPackage rec {
      pname = "transformers-stream-generator";
      version = "0.0.5";
      format = "setuptools";

      src = super.fetchFromGitHub {
        owner = "LowinLi";
        repo = "transformers-stream-generator";
        rev = "main";
        sha256 = "sha256-Jn1zD/Pm2BysCM9ZLHGpWtAwBYt3kcEljCPFyloDiT8=";
      };

      propagatedBuildInputs = with super.python3Packages; [
        transformers
        # Use torch from environment
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "A text generation method which returns a generator, streaming out each token in real-time during inference";
        homepage = "https://github.com/LowinLi/transformers-stream-generator";
        license = licenses.mit;
      };
    };
  };
}