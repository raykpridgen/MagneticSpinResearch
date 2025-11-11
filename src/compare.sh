#!/bin/bash

# Number of iterations
NUM_RUNS=10

# Make files
n=1024
i=2
echo "Making files for computation"
while [ $i -le $n ]
do
    Afile="../data/test_items/A${i}.csv"
    echo "${i} ${i}" > "$Afile"
    awk -v size=$i '
    BEGIN {
        srand()
        for (r = 1; r <= size; r++) {
            for (c = 1; c <= size; c++) {
                printf "%.3f", rand()
                if (c < size) printf " "
            }
            printf "\n"
        }
    }' >> "$Afile"

    # ---- Create B{i}.csv ----
    Bfile="../data/test_items/B${i}.csv"
    echo "${i} 1" > "$Bfile"
    awk -v size=$i '
    BEGIN {
        srand()
        for (r = 1; r <= size; r++) {
            printf "%.3f\n", rand()
        }
    }' >> "$Bfile"

    # Increment i by powers of 2
    i=$(( i * 2 ))
done

echo "Compiling programs"
g++ solve.cpp -o solve_cpu 

# Paths to your programs
PROG1="./solve_cpu"
PROG2="./solve_gpu"
> ../data/timing.txt 

echo "Computing systems"
# Loop over counter
for ((i=1; i<=NUM_RUNS; i++))
do
    RUN_SIZE=$((2 ** i))
    echo "Solving system of $RUN_SIZE degree"

    # Example: pass the counter as a command-line argument
    $PROG1 "../data/test_items/A$RUN_SIZE.csv" "../data/test_items/B$RUN_SIZE.csv" "none"
    if [ $? -ne 0 ]; then
        echo "Program1 failed on run $i"
        exit 1
    fi

    # Run the second program with the same counter
    #$PROG2 "../data/A$i.csv" "../data/B$i.csv" "none"
    #if [ $? -ne 0 ]; then
    #    echo "Program2 failed on run $i"
    #    exit 1
    #fi

    echo "Run #$i complete"
done

echo "All runs finished!"
