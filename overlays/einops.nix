self: super:
let
  inherit (super) lib;

  addEinopsOverrides = pkg:
    pkg.overrideAttrs (old: {
      doCheck = false;
      checkInputs = [ ];
      disabledTestPaths = (old.disabledTestPaths or []) ++ [ "scripts/test_notebooks.py" ];
      disabledTests = (old.disabledTests or []) ++ [ "test_notebook_3" ];
    });

  overrideEinops = pkgSet:
    pkgSet // {
      einops = addEinopsOverrides pkgSet.einops;
    };

  extendPython = python:
    let
      existing = python.packageOverrides or (_: _: {});
      composed = lib.composeExtensions existing (_final: prev: {
        einops = addEinopsOverrides prev.einops;
      });
    in python.override { packageOverrides = composed; };

in (
  {
    python3 = extendPython super.python3;
    python3Packages = overrideEinops super.python3Packages;
  }
  // lib.optionalAttrs (super ? python313) {
    python313 = extendPython super.python313;
    python313Packages = overrideEinops super.python313Packages;
  }
)
