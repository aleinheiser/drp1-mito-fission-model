{
  buildPythonPackage,
  fetchPypi,
  # Dependencies
  numpy,
  scipy,
  matplotlib,
  pandas,
  multiprocess,
}: let
  pname = "salib";
  version = "1.4.8";
  src = fetchPypi {
    inherit pname version;
    dist = "py3";
    python = "py3";
    format = "wheel";
    hash = "sha256-/vDwY3qX8a3KNaQ15BJ0rdYnXqQkKPBNPoPetyDMAjQ=";
  };
in
  buildPythonPackage {
    inherit pname version;

    inherit src;

    format = "wheel";

    propagatedBuildInputs = [
      numpy
      scipy
      matplotlib
      pandas
      multiprocess
    ];

    pythonImportsCheck = ["SALib"];
  }
