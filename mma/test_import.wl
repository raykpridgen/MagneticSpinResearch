(*
   Test script to demonstrate importing C++ solver results into Mathematica
*)

Print["==========================================="]
Print["Testing C++ Matrix Solver Import"]
Print["==========================================="]
Print[""]

(* Import the solution file *)
solutionFile = FileNameJoin[{DirectoryName[$InputFileName], "..", "data", "mma_out", "matrix_data_1_solution.txt"}];

Print["Importing solution from: ", solutionFile]
Print[""]

(* Method 1: Using Get[] - simplest method *)
solution = Get[solutionFile];

Print["Solution imported successfully!"]
Print["Dimensions: ", Dimensions[solution]]
Print["First 5 elements: ", solution[[1;;5]]]
Print[""]

(* Verify it's a proper list of complex numbers *)
Print["Type check:"]
Print["  Is list? ", ListQ[solution]]
Print["  First element: ", solution[[1]], " (type: ", Head[solution[[1]]], ")"]
Print["  Second element: ", solution[[2]], " (type: ", Head[solution[[2]]], ")"]
Print[""]

(* Calculate some properties *)
Print["Solution properties:"]
Print["  Norm: ", Norm[solution]]
Print["  Max magnitude: ", Max[Abs[solution]]]
Print["  Number of real elements: ", Count[solution, _Real]]
Print["  Number of complex elements: ", Count[solution, _Complex]]
Print[""]

(* Method 2: Using Import[] - also works *)
solution2 = ToExpression[Import[solutionFile, "String"]];
Print["Verification - both methods give same result: ", solution === solution2]
Print[""]

Print["Import test completed successfully!"]
