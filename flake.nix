{
  description = "An awesome machine-learning project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";

    utils.url = "github:numtide/flake-utils";

    ml-pkgs.url = "github:nixvital/ml-pkgs";
    ml-pkgs.inputs.nixpkgs.follows = "nixpkgs";
    ml-pkgs.inputs.utils.follows = "utils";
  };

  outputs = { self, nixpkgs, ... }@inputs: {
    overlays.dev = nixpkgs.lib.composeManyExtensions [
      inputs.ml-pkgs.overlays.jax-family
    ];
  } // inputs.utils.lib.eachSystem [
    "x86_64-linux"
  ] (system:
    let 
       bitarray = pkgs.python310Packages.buildPythonPackage rec {
           pname = "bitarray";
           version = "2.9.2";
           format = "setuptools";
           src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "a8f286a51a32323715d77755ed959f94bef13972e9a2fe71b609e40e6d27957e";
           };
           nativeBuildInputs = [
             pkgs.python3
             pkgs.buildPackages.python310Packages.cffi  # Example dependency for CFFI-based extensions
             pkgs.buildPackages.python310Packages.setuptools
             pkgs.buildPackages.python310Packages.setuptools-scm
             pkgs.buildPackages.python310Packages.wheel
           ];
           setupPyBuildFlags = [ "--inplace" ];
        };

        dfa = pkgs.python310Packages.buildPythonPackage rec {
          pname = "dfa";
          version = "4.6.3";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "b4d511f73eb1588a391cc4a032362c836053963a17aa4bc58b40c59a39ea639a";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python310Packages.attrs
            pkgs.python310Packages.funcy
            pkgs.python310Packages.pydot
            pkgs.python310Packages.bidict
            bitarray
          ];
        };


        dfa-mutate = pkgs.python310Packages.buildPythonPackage rec {
          pname = "dfa_mutate";
          version = "0.1.3";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "abe00e8afb4b7bf164806d3696caa1e511b2e869ac272f22b8e8afee4018d5ed";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [ dfa ];
        };

        dfa-identify = pkgs.python310Packages.buildPythonPackage rec {
          pname = "dfa_identify";
          version = "3.13.0";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "4e701e25782d87ccf9c0d68f7d4bdac0f123d2bbed4c8dee1c0a2c706f5f6038";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            dfa
            pkgs.python310Packages.attrs
            pkgs.python310Packages.bidict
            pkgs.python310Packages.funcy
            pkgs.python310Packages.more-itertools
            pkgs.python310Packages.networkx
            pkgs.python310Packages.python-sat
          ];
        };

        jraph = pkgs.python310Packages.buildPythonPackage rec {
          pname = "jraph";
          version = "0.0.6.dev0";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "350fe37bf717f934f1f84fd3370a480b3178bfcb61dfa217c738971308c57625";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python310Packages.jax
            pkgs.python310Packages.jaxlib-bin
            pkgs.python310Packages.numpy
          ];
        };

        rotary-embedding-torch = pkgs.python310Packages.buildPythonPackage rec {
          pname = "rotary_embedding_torch";
          version = "0.5.3";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "7c0297a21301fbd0d20cfdcba33be5d1ad0a46e33454168af08413dbf3e07457";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python310Packages.einops
            pkgs.python310Packages.torch
            pkgs.python310Packages.beartype
          ];
        };


        graph-transformer-pytorch = pkgs.python310Packages.buildPythonPackage rec {
          pname = "graph_transformer_pytorch";
          version = "0.1.1";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "e1dd4761c9f944362a9db52454512614b93b47182e57b888761cffee17973eb6";
            dist = python;
            python = "py3";
          };
          propagatedBuildInputs = [
            pkgs.python310Packages.torch
            pkgs.python310Packages.einops
            rotary-embedding-torch
          ];
        };

        # Maybe a typo in the graphtranformers library
        einsum = pkgs.python310Packages.buildPythonPackage rec {
          pname = "einsum";
          version = "0.3.0";
          format = "wheel";
          src = pkgs.python310Packages.fetchPypi rec {
            inherit pname version format;
            sha256 = "6d2b1f7932879d630969bfb599e09f8448dcbc38a2be6cee192b8a1ae02c52dc";
            dist = python;
            python = "py3";
          };
        };

        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            cudaCapabilities = [ "7.5" "8.6" ];
            cudaForwardCompat = false;
          };
          overlays = [ self.overlays.dev ];
        };
    in {
      devShells.default = let
        python-env = pkgs.python310.withPackages (pyPkgs: with pyPkgs; [
          numpy
          scipy
          pandas
          dfa
          dfa-identify
          dfa-mutate
          #jax
          #jaxlib-bin
          #jraph
          torch
          graph-transformer-pytorch
          tqdm
          # pytorch-lightning
        ]);

        name = "jax-equinox-basics";
      in pkgs.mkShell {
        inherit name;

        packages = [
          python-env
          pkgs.python310Packages.flit
          pkgs.python310Packages.ptpython
          pkgs.python310Packages.jupyterlab
          pkgs.python310Packages.jupytext
          pkgs.graphviz
        ];
      };
    });
}
