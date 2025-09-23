final: prev: {
  python3Packages = prev.python3Packages // {
    triton = prev.python3Packages.triton.overrideAttrs (oldAttrs: {
      buildInputs = (oldAttrs.buildInputs or []) ++ [
        final.cmake
        final.ninja
        final.llvm_18
        final.zlib
        final.ncurses
      ];
      
      nativeBuildInputs = (oldAttrs.nativeBuildInputs or []) ++ [
        final.cmake
        final.ninja
        final.pkg-config
        final.python3Packages.pybind11
      ];

      # Fix CMake configuration for triton
      preConfigure = ''
        # Conservative parallel job settings to prevent OOM
        export CMAKE_BUILD_PARALLEL_LEVEL=4
        export MAX_JOBS=4
        export MAKEFLAGS="-j4"
        
        # Fix LLVM paths
        export LLVM_INCLUDE_DIRS=${final.llvm_18.dev}/include
        export LLVM_LIBRARY_DIRS=${final.llvm_18.lib}/lib
        export LLVM_TARGETS_TO_BUILD="X86;NVPTX"
        
        # Set CUDA architecture for triton
        export TORCH_CUDA_ARCH_LIST="8.6"
        export CUDA_ARCH_LIST="8.6"
        
        # Memory limits
        export CXXFLAGS="$CXXFLAGS -O2"
        export CFLAGS="$CFLAGS -O2"
        
        # Disable problematic features
        export TRITON_DISABLE_LINE_INFO=1
      '' + (oldAttrs.preConfigure or "");

      # Override CMake flags for better compatibility
      cmakeFlags = [
        "-DCMAKE_BUILD_TYPE=Release"
        "-DTRITON_BUILD_TUTORIALS=OFF"
        "-DTRITON_BUILD_PYTHON_MODULE=ON"
        "-DLLVM_INCLUDE_DIRS=${final.llvm_18.dev}/include"
        "-DLLVM_LIBRARY_DIRS=${final.llvm_18.lib}/lib"
        "-DCMAKE_CXX_STANDARD=17"
        "-DLLVM_TARGETS_TO_BUILD=X86;NVPTX"
        "-DCMAKE_BUILD_PARALLEL_LEVEL=4"
        "-DLLVM_PARALLEL_LINK_JOBS=2"
      ] ++ (oldAttrs.cmakeFlags or []);

      # Resource limits
      requiredSystemFeatures = [ "big-parallel" ];
      
      # Add timeout for long builds
      env = (oldAttrs.env or {}) // {
        NIX_BUILD_CORES = "4";
        MAKEFLAGS = "-j4";
      };
      
      # Skip tests to save build time and memory
      doCheck = false;
      dontUsePytestCheck = true;
    });
  };
}