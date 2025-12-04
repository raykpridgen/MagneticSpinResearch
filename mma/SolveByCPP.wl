(* ::Package:: *)

(*
   Mathematica Helper Functions for C++ Matrix Solver Integration

   This package provides functions to export matrices from Mathematica,
   solve them using the C++ matrix solver (with CUDA support),
   and import the results back into Mathematica.
*)

BeginPackage["SolveByCPP`"]

ExportMatrixForCPP::usage = "ExportMatrixForCPP[matrix, filename] exports a matrix to a format readable by the C++ solver."

SolveMatrixCPP::usage = "SolveMatrixCPP[matrix, useCUDA:False] exports matrix, calls C++ solver, and imports solution."

Begin["`Private`"]

(* Get the base directory where the spins project is located *)
$SpinsDir = DirectoryName[DirectoryName[$InputFileName], 1];

(* Export matrix to C++ readable format *)
ExportMatrixForCPP[matrix_?MatrixQ, filename_String] := Module[
  {outputPath, formatMatrix},

  (* Create output path *)
  outputPath = FileNameJoin[{$SpinsDir, "data", "mma_out", filename}];

  (* Ensure directory exists *)
  If[!DirectoryQ[DirectoryName[outputPath]],
    CreateDirectory[DirectoryName[outputPath]]
  ];

  (* Format matrix with Complex notation for C++ *)
  formatMatrix = Map[
    If[Im[#] == 0,
      N[Re[#]],
      "Complex(" <> ToString[N[Re[#]]] <> ", " <> ToString[N[Im[#]]] <> ")"
    ] &,
    matrix,
    {2}
  ];

  (* Export as nested list *)
  Export[outputPath, formatMatrix, "Text"];

  Print["Matrix exported to: ", outputPath];
  outputPath
]

(* Import solution from C++ solver *)
ImportSolutionFromCPP[solutionFile_String] := Module[
  {solution},

  (* Read and evaluate the file - it's in native Mathematica format *)
  solution = Get[solutionFile];

  (* Return the solution *)
  solution
]

(* Main solver function *)
SolveMatrixCPP[matrix_?MatrixQ, useCUDA_:False] := Module[
  {tempFile, outputFile, solverPath, command, result, solution},

  (* Check if matrix is square *)
  If[Dimensions[matrix][[1]] != Dimensions[matrix][[2]],
    Message[SolveMatrixCPP::nonsquare, "Matrix must be square"];
    Return[$Failed]
  ];

  (* Generate temporary filename *)
  tempFile = "temp_matrix_" <> CreateUUID[] <> ".txt";

  (* Export matrix *)
  ExportMatrixForCPP[matrix, tempFile];

  (* Determine solver path *)
  solverPath = FileNameJoin[{$SpinsDir, "build",
    If[useCUDA, "matrix_solver_cuda", "matrix_solver"]}];

  (* Check if solver exists *)
  If[!FileExistsQ[solverPath],
    Print["Error: Solver not found at: ", solverPath];
    Print["Please build the solver first using 'make matrix-solver' or 'make matrix-solver-cuda'"];
    Return[$Failed]
  ];

  (* Construct command *)
  command = solverPath <> " " <>
    FileNameJoin[{$SpinsDir, "data", "mma_out", tempFile}] <>
    If[useCUDA, " --cuda", ""];

  Print["Running C++ solver..."];
  Print["Command: ", command];

  (* Run solver *)
  result = Run[command];

  If[result != 0,
    Print["Error: Solver returned error code ", result];
    Return[$Failed]
  ];

  (* Import solution *)
  outputFile = FileNameJoin[{$SpinsDir, "data", "mma_out",
    StringReplace[tempFile, ".txt" -> "_solution.txt"]}];

  If[!FileExistsQ[outputFile],
    Print["Error: Solution file not found: ", outputFile];
    Return[$Failed]
  ];

  solution = ImportSolutionFromCPP[outputFile];

  Print["Solution imported successfully!"];
  Print["Solution dimension: ", Length[solution]];

  (* Clean up temporary files *)
  Quiet[DeleteFile[FileNameJoin[{$SpinsDir, "data", "mma_out", tempFile}]]];
  Quiet[DeleteFile[outputFile]];

  solution
]

End[]
EndPackage[]

(* Usage Examples *)
Print["========================================"]
Print["C++ Matrix Solver Integration Loaded"]
Print["========================================"]
Print[""]
Print["Available functions:"]
Print["  ExportMatrixForCPP[matrix, filename] - Export matrix for C++ solver"]
Print["  SolveMatrixCPP[matrix, useCUDA]      - Solve matrix using C++ (CPU or CUDA)"]
Print[""]
Print["Example usage:"]
Print["  (* Solve using CPU *)"]
Print["  solution = SolveMatrixCPP[myMatrix, False]"]
Print[""]
Print["  (* Solve using CUDA GPU *)"]
Print["  solution = SolveMatrixCPP[myMatrix, True]"]
Print[""]
