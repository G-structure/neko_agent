self: super: {
  python3Packages = super.python3Packages // {
    einops = super.python3Packages.einops.overridePythonAttrs (old: {
      # Skip tests to avoid timeout issues in Jupyter notebook tests
      doCheck = false;
      checkInputs = [ ];
    });
  };
}