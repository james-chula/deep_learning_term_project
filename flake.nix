{
  description = "BloodMNIST ResNet-18 project development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          python = pkgs.python312;
          pythonEnv = python.withPackages (ps: with ps; [
            ipykernel
            jupyterlab
            matplotlib
            numpy
            pandas
            pillow
            scikit-learn
            scipy
            seaborn
            torch
            torchvision
            tqdm
          ]);
        in {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.git
              pkgs.uv
            ];

            # Keep Jupyter resources, configuration, and transient runtime data
            # inside this repository rather than in the user's home directory.
            JUPYTER_PATH = "${toString ./.}/.jupyter/share/jupyter";
            JUPYTER_CONFIG_DIR = "${toString ./.}/.jupyter/config";
            JUPYTER_DATA_DIR = "${toString ./.}/.jupyter/data";
            JUPYTER_RUNTIME_DIR = "${toString ./.}/.jupyter/runtime";
            PYTHONNOUSERSITE = "1";

            shellHook = ''
              export PROJECT_ROOT="$PWD"
              export VIRTUAL_ENV_DISABLE_PROMPT=1
              echo "BloodMNIST environment loaded. Run: ./scripts/setup-runtime"
              echo "Then start Jupyter with: ./scripts/jupyter"
            '';
          };
        });
    };
}
