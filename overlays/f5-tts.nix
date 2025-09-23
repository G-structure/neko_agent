self: super: {
  python3Packages = super.python3Packages // {
    f5-tts = super.python3Packages.buildPythonPackage rec {
      pname = "f5-tts";
      version = "1.1.7";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "SWivid";
        repo = "F5-TTS";
        rev = "main";
        sha256 = "sha256-6IRJ+x5Pu/7FylZE7BReH0HnzRkG97luNd909LrR4RU=";
      };

      nativeBuildInputs = with super.python3Packages; [
        setuptools
        setuptools-scm
        wheel
      ];

      propagatedBuildInputs = with super.python3Packages; [
        (accelerate.override { torch = super.python3Packages."torch-bin"; })
        cached-path
        click
        datasets
        ema-pytorch
        gradio
        hydra-core
        jieba
        librosa
        matplotlib
        pydantic
        pydub
        pypinyin
        safetensors
        soundfile
        tomli
        super.python3Packages."torchaudio-bin"
        (torchdiffeq.override { torch = super.python3Packages."torch-bin"; })
        transformers
        transformers-stream-generator
        tqdm
        unidecode
        vocos
        wandb
        (x-transformers.override { torch = super.python3Packages."torch-bin"; })
        # Basic dependencies
        requests
        setuptools
        wheel
      ] ++ super.lib.optionals (!super.stdenv.isAarch64 && !super.stdenv.isDarwin) [
        super.python3Packages.bitsandbytes
      ];

      # Override version constraints that are too restrictive
      postPatch = ''
        # Remove restrictive version constraints
        substituteInPlace pyproject.toml \
          --replace "numpy<=1.26.4" "numpy" \
          --replace "pydantic<=2.10.6" "pydantic"
      '';

      # Skip dependency check for packages we'll add later
      pythonImportsCheck = [ ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching";
        homepage = "https://github.com/SWivid/F5-TTS";
        license = licenses.mit;
        platforms = platforms.linux ++ platforms.darwin;
      };
    };
  };
}
