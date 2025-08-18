self: super: {
  python3Packages = super.python3Packages // {
    streaming = super.python3Packages.buildPythonPackage rec {
      pname = "mosaicml-streaming";
      version = "0.13.0";
      format = "setuptools";

      src = super.fetchPypi {
        pname = "mosaicml_streaming";
        inherit version;
        sha256 = "sha256-CYGeJYLHcc2lP0tVGOOWjQJAdW5Y6Rzb2KRu28icgxw=";
      };

      propagatedBuildInputs = with super.python3Packages; [
        numpy
        # torch  # Use the torch provided by the environment
        psutil
        pyyaml
        requests
        tqdm
        xxhash
        zstandard
        brotli
        python-snappy
        pyzstd
        zstd
        catalogue
      ];

      doCheck = false;  # Skip tests to avoid potential issues

      meta = with super.lib; {
        description = "MosaicML Streaming - efficient data loading for ML training";
        homepage = "https://github.com/mosaicml/streaming";
        license = licenses.asl20;
      };
    };
  };
}
