# Plan for this project -- Modelling mathematica code

# Workflow, modules / components needed

## 1. Base Spin Matrices - Complete
- Array of doubles
- MAYBE need function to compute these for different patterns
- Keep row major / col major in mind
 
### Consisting of:
- sx: 1, 2      (4 x 4)
- sy: 1, 2      (4 x 4)
- sz: 1, 2      (4 x 4)

## 2. Lifted Operators: Kronecker Product - Complete
- Kronecker already implemented in kronecker.cu - unneeded?

### With function, generate 8x8 matrices:
- Sx 1, 2 = kronecker(sx, Identity[2])              (8 x 8)
- Sy 1, 2 = kronecker(sy, Identity[2])              (8 x 8)
- Sz 1, 2 = kronecker(sz, Identity[2])              (8 x 8)
- Ix, y, z = kronecker(Identity[4], 1/2 PauliMatrix[1])   (8 x 8)

## 3. Construct Numerical Hamiltonian

### Sum:
- Sz1 + Sz2

### Compute Products:
- Ix * Sx1
- Iy * Sy1
- Iz * Sz1

### Scale by constants

### Combine into single 8x8 matrix H

## 4. Build Liouvillian Superoperator

### Commutator
- [H, p] = Hp - pH

### Anticommutator
- {P, p} = Pp + pP

### Projection operators
- Ps, Pt
- Diagonal matrices

### Scalar coefficients

### Vectorization helpers
- Flatten 8x8 to 64x1

### Build 64x64 L representing SLE operator

## 5. Solve Steady State Equation

### Form equation Lp = b

### Replace a row of L with a trace constraint:
- i -> SUM(pii) = 1

### Set RHS = 1

### Solve the 64x64 system

### Returns p for a given Bz

## 6. Extract observable
 
### Extract
- Ps * p (8 x 8)

### Store
- (Bz, Value)



## 7. Sweep over Bz

### Loop
- Increment by step
- Bz min - max

### Execute
- build H
- build L
- solve rho
- compute observable
- store in vector

## Write to CSV
- Vector of (Bz, value)'s 