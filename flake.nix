{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = {
    self,
    nixpkgs,
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    python = pkgs.python311.withPackages (ps: [
      ps.autograd
      ps.joblib
      ps.matplotlib
      ps.numpy
      ps.scipy
      ps.tqdm
      (ps.callPackage ./nix/salib.nix {})
    ]);
  in {
    formatter.${system} = pkgs.alejandra;

    devShells.${system}.default = pkgs.mkShell {
      buildInputs = [
        python
        pkgs.ruff
        pkgs.inkscape
        pkgs.imagemagick
        pkgs.zip
      ];
      shellHook = ''
        export DATAPATH=$(realpath data)
        export PYTHONPATH=$(realpath code):$PYTHONPATH
      '';
    };
  };
}
