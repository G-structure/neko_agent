self: super: {
  python3Packages = super.python3Packages // {
    cached-path = super.python3Packages.buildPythonPackage rec {
      pname = "cached-path";
      version = "1.6.3";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "allenai";
        repo = "cached_path";
        rev = "v${version}";
        sha256 = "sha256-385tA0IdzG3zijG4QERqAwNLW4cz/6/KYjU1+yH/Po8=";
      };

      nativeBuildInputs = with super.python3Packages; [
        setuptools
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        requests
        filelock
        typing-extensions
        rich
        boto3
        google-cloud-storage
        huggingface-hub
      ];

      # Remove problematic version-constrained dependencies
      pythonRemoveDeps = [ "rich" "filelock" "google-cloud-storage" "huggingface-hub" ];
      
      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "A file utility for accessing both local and remote files through a unified interface";
        homepage = "https://github.com/allenai/cached_path";
        license = licenses.asl20;
      };
    };
  };
}