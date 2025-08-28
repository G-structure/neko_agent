# Nix Flake Python Packaging Guide

This guide explains the nuances and patterns used for packaging Python dependencies in the Neko Agent project's Nix flake. The project uses a sophisticated overlay system to package complex Python libraries that aren't available in standard Nixpkgs, with special attention to PyTorch ecosystem integration and CUDA support.

## Overview

The Neko Agent project requires numerous specialized Python packages for AI/ML, WebRTC, and audio processing that either don't exist in Nixpkgs or need custom configurations. The flake uses a comprehensive overlay system to package these dependencies while maintaining compatibility with the PyTorch ecosystem.

### Key Packaging Principles

1. **Torch-bin Integration** - Use pre-compiled PyTorch wheels for consistency
2. **CUDA Compatibility** - Ensure proper CUDA toolkit integration
3. **Dependency Override** - Override transitive dependencies to use `torch-bin`
4. **Test Skipping** - Disable problematic tests that require GPU access
5. **Version Flexibility** - Patch restrictive version constraints when needed

## Core Patterns

### 1. Basic Python Package Overlay

```nix
self: super: {
  python3Packages = super.python3Packages // {
    example-package = super.python3Packages.buildPythonPackage rec {
      pname = "example-package";
      version = "1.0.0";
      format = "pyproject";

      src = super.fetchFromGitHub {
        owner = "owner";
        repo = "example-package";
        rev = "v${version}";
        sha256 = "...";
      };

      nativeBuildInputs = with super.python3Packages; [ setuptools wheel ];
      propagatedBuildInputs = with super.python3Packages; [ numpy scipy ];
      doCheck = false;

      meta = with super.lib; {
        description = "Package description";
        homepage = "https://github.com/owner/example-package";
        license = licenses.mit;
      };
    };
  };
}
```

### 2. PyTorch Integration Pattern

Always use `torch-bin` and override transitive dependencies:

```nix
self: super: {
  python3Packages = super.python3Packages // {
    ml-package = super.python3Packages.buildPythonPackage rec {
      # ... basic fields ...
      
      propagatedBuildInputs = with super.python3Packages; [
        super.python3Packages."torch-bin"
        super.python3Packages."torchvision-bin"
        
        # Override packages that depend on torch
        (accelerate.override { torch = super.python3Packages."torch-bin"; })
        (torchdiffeq.override { torch = super.python3Packages."torch-bin"; })
        
        transformers
        numpy
      ];

      doCheck = false;  # Skip GPU-dependent tests
    };
  };
}
```

### 3. CUDA-Aware Package Pattern

For packages needing CUDA compilation:

```nix
self: super: 
let
  cudaPackages = super.cudaPackages_12_8;
  cuda-redist = super.symlinkJoin {
    name = "cuda-redist-${cudaPackages.cudaMajorMinorVersion}";
    paths = with cudaPackages; [
      (super.lib.getDev cuda_cccl)
      (super.lib.getDev libcublas)
      # ... more CUDA libs
    ];
  };
in
{
  python3Packages = super.python3Packages // {
    cuda-package = super.python3Packages.buildPythonPackage rec {
      # ... basic fields ...
      
      postPatch = ''
        substituteInPlace setup.py \
          --replace-fail "find_cuda_libs()" "['${cuda-redist}/lib/libcudart.so']"
      '';

      nativeBuildInputs = [ super.cmake cudaPackages.cuda_nvcc ];
      buildInputs = [ cuda-redist ];
      
      CUDA_HOME = "${cuda-redist}";
      NVCC_PREPEND_FLAGS = [ "-I${cuda-redist}/include" ];
      
      doCheck = false;
    };
  };
}
```

### 4. Version Constraint Patching

Relax overly restrictive version constraints:

```nix
postPatch = ''
  substituteInPlace pyproject.toml \
    --replace "numpy<=1.24.0" "numpy" \
    --replace "pydantic>=2.0,<2.5" "pydantic>=2.0"
'';
## Real-World Examples

### F5-TTS Package

Demonstrates torch overrides, version patching, and platform-specific dependencies:

```nix
self: super: {
  python3Packages = super.python3Packages // {
    f5-tts = super.python3Packages.buildPythonPackage rec {
      pname = "f5-tts";
      version = "1.1.7";
      
      src = super.fetchFromGitHub {
        owner = "SWivid";
        repo = "F5-TTS";
        rev = "main";
        sha256 = "sha256-MtPyqS5aNrq929pHMlDp3HFUSf+i9xYDb5xMA0Eqh9Y=";
      };

      propagatedBuildInputs = with super.python3Packages; [
        # Torch overrides
        (accelerate.override { torch = super.python3Packages."torch-bin"; })
        (x-transformers.override { torch = super.python3Packages."torch-bin"; })
        
        # Custom overlays
        cached-path ema-pytorch vocos
        
        # Standard deps
        transformers gradio librosa soundfile
      ] ++ super.lib.optionals (!super.stdenv.isAarch64) [
        bitsandbytes  # x86_64 only
      ];

      # Version constraint fixes
      postPatch = ''
        substituteInPlace pyproject.toml \
          --replace "numpy<=1.26.4" "numpy" \
          --replace "pydantic<=2.10.6" "pydantic"
      '';
      
      doCheck = false;
    };
  };
}
```

### Pi-Zero PyTorch

Shows complex dependency chains and overlay usage:

```nix
self: super: {
  python3Packages = super.python3Packages // {
    pi-zero-pytorch = super.python3Packages.buildPythonPackage rec {
      pname = "pi-zero-pytorch";
      version = "0.2.5";
      
      src = super.fetchFromGitHub {
        owner = "lucidrains";
        repo = "pi-zero-pytorch";
        rev = "1c13cbbcc236c3cb4fd18213c543188fdf083b33";
        sha256 = "1xl38c5spv798xydljahq8fw9nglwcw95s95zpn11cm9ra4rg4ib";
      };

      nativeBuildInputs = [ super.python3Packages.hatchling ];
      
      propagatedBuildInputs = with super.python3Packages; [
        # Torch overrides
        (accelerate.override { torch = super.python3Packages."torch-bin"; })
        
        # Custom overlays (all from overlays/)
        einx x-transformers rotary-embedding-torch
        accelerated-scan bidirectional-cross-attention
        
        # Use patched einops from self
        (self.python3Packages.einops)
        
        beartype jaxtyping scipy tqdm
        super.python3Packages."torch-bin"
      ];
      
      doCheck = false;
    };
  };
}
```

## Advanced Techniques

### Using `self` vs `super`

- **`super`**: Access packages before overlay modifications
- **`self`**: Access final overlaid packages (avoid infinite recursion)

```nix
propagatedBuildInputs = with super.python3Packages; [
  numpy  # Use super for standard packages
  (self.python3Packages.custom-overlay-package)  # Use self for overlay deps
  (accelerate.override { torch = super.python3Packages."torch-bin"; })
];
```

### Conditional Dependencies

```nix
propagatedBuildInputs = with super.python3Packages; [
  numpy scipy
] ++ super.lib.optionals (!super.stdenv.isAarch64) [
  bitsandbytes  # x86_64 only
] ++ super.lib.optionals super.stdenv.isLinux [
  nvidia-ml-py  # Linux only
];
```

### Source Fetching

```nix
# Preferred: Specific tag/commit
src = super.fetchFromGitHub {
  owner = "owner"; repo = "repo";
  tag = "v${version}";  # or rev = "commit-hash";
  hash = "sha256-...";  # Use nix-prefetch-github
};

# PyPI when available
src = super.fetchPypi {
  inherit pname version;
  hash = "sha256-...";
};
```

### Build Systems

```nix
# Modern pyproject.toml
format = "pyproject";
nativeBuildInputs = [ super.python3Packages.hatchling ];

# Legacy setup.py  
format = "setuptools";
nativeBuildInputs = [ super.python3Packages.setuptools ];
```

## Best Practices

### Hash Management
```bash
# Get hashes for sources
nix-prefetch-github owner repo --rev v1.0.0
nix hash to-sri --type sha256 abc123...  # Convert to SRI format
```

### Testing Overlays
```bash
# Test single package
nix-shell -p '(python3.withPackages (ps: [ ps.my-package ]))'
nix-shell --run 'python -c "import my_package"'
```

### Dependency Management
- Keep dependencies minimal and explicit
- Always use `torch-bin` for PyTorch ecosystem
- Override transitive torch dependencies

### Quality Metadata
```nix
meta = with super.lib; {
  description = "Clear, concise description";
  homepage = "https://github.com/owner/repo";
  license = licenses.mit;
  platforms = platforms.unix;
};
```

## Troubleshooting

### Common Issues

**Import Errors**: Use `super.python3Packages."torch-bin"` not `"torch"`

**Version Conflicts**: Patch constraints with `postPatch`
```nix
postPatch = ''
  substituteInPlace setup.py --replace "torch==2.0.0" "torch>=2.0.0"
'';
```

**CUDA Issues**: Set proper CUDA environment
```nix
buildInputs = [ cuda-redist ];
CUDA_HOME = "${cuda-redist}";
```

**Test Failures**: Disable with `doCheck = false;`

## Integration with Flake

### Adding New Overlays

1. Create overlay file in `overlays/` directory
2. Add to `nekoOverlays` list in `flake.nix`
3. Include in Python environments

```nix
# In flake.nix
nekoOverlays = [
  (import ./overlays/new-package.nix)
];

# Usage in environments
pyEnvExample = pkgs.python3.withPackages (ps: [
  ps.new-package
]);
```

### Overlay Ordering

Dependencies must come first:

```nix
nekoOverlays = [
  # Base packages first
  (import ./overlays/cached-path.nix)
  (import ./overlays/ema-pytorch.nix)
  
  # Dependent packages after
  (import ./overlays/f5-tts.nix)  # Uses ema-pytorch
  
  # Complex packages last
  (import ./overlays/pi-zero-pytorch/pi-zero-pytorch.nix)
  
  # Patches to existing packages last
  (import ./overlays/einops.nix)  # Overrides existing einops
];
```

This guide covers the key patterns for packaging Python dependencies in the Neko Agent project with proper PyTorch ecosystem integration and CUDA support.