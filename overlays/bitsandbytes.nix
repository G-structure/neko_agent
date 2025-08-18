self: super: 
let
  torch = super.python3Packages."torch-bin";
  cudaPackages = super.cudaPackages_12_8;
  cudaSupport = true;
  inherit (cudaPackages) cudaMajorMinorVersion;
  
  cudaMajorMinorVersionString = super.lib.replaceStrings [ "." ] [ "" ] cudaMajorMinorVersion;

  # CUDA redistribution packages
  cuda-common-redist = with cudaPackages; [
    (super.lib.getDev cuda_cccl)
    (super.lib.getDev libcublas)
    (super.lib.getLib libcublas)
    libcurand
    libcusolver
    (super.lib.getDev libcusparse)
    (super.lib.getLib libcusparse)
    (super.lib.getDev cuda_cudart)
  ];

  cuda-native-redist = super.symlinkJoin {
    name = "cuda-native-redist-${cudaMajorMinorVersion}";
    paths = with cudaPackages; [
      (super.lib.getDev cuda_cudart)
      (super.lib.getLib cuda_cudart)
      (super.lib.getStatic cuda_cudart)
      cuda_nvcc
    ] ++ cuda-common-redist;
  };

  cuda-redist = super.symlinkJoin {
    name = "cuda-redist-${cudaMajorMinorVersion}";
    paths = cuda-common-redist;
  };
in
{
  python3Packages = super.python3Packages // {
    bitsandbytes = super.python3Packages.buildPythonPackage rec {
      pname = "bitsandbytes";
      version = "0.46.0";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "bitsandbytes-foundation";
        repo = "bitsandbytes";
        tag = version;
        hash = "sha256-q1ltNYO5Ex6F2bfCcsekdsWjzXoal7g4n/LIHVGuj+k=";
      };

      postPatch = super.lib.optionalString cudaSupport ''
        substituteInPlace bitsandbytes/cextension.py \
          --replace-fail "if cuda_specs:" "if True:" \
          --replace-fail \
            "cuda_binary_path = get_cuda_bnb_library_path(cuda_specs)" \
            "cuda_binary_path = PACKAGE_DIR / 'libbitsandbytes_cuda${cudaMajorMinorVersionString}.so'"
      '';

      nativeBuildInputs = with super; [
        cmake
        cudaPackages.cuda_nvcc
      ];

      build-system = with super.python3Packages; [
        setuptools
      ];

      buildInputs = super.lib.optionals cudaSupport [ cuda-redist ];

      cmakeFlags = [
        (super.lib.cmakeFeature "COMPUTE_BACKEND" (if cudaSupport then "cuda" else "cpu"))
      ];
      
      CUDA_HOME = "${cuda-native-redist}";
      NVCC_PREPEND_FLAGS = super.lib.optionals cudaSupport [
        "-I${cuda-native-redist}/include"
        "-L${cuda-native-redist}/lib"
      ];

      preBuild = ''
        make -j $NIX_BUILD_CORES
        cd .. # leave /build/source/build
      '';

      dependencies = with super.python3Packages; [
        scipy
        torch
      ];

      doCheck = false; # tests require CUDA and GPU access

      pythonImportsCheck = [ "bitsandbytes" ];

      meta = with super.lib; {
        description = "8-bit CUDA functions for PyTorch";
        homepage = "https://github.com/bitsandbytes-foundation/bitsandbytes";
        changelog = "https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/${version}";
        license = licenses.mit;
        maintainers = with maintainers; [ bcdarwin ];
      };
    };
  };
}