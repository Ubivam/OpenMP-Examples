# C compiler
CC = gcc
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) jacobi.c -o jacobi -lm 
	$(CC)  jacobi_seq.c -o jacobi_seq -lm
clean:
	rm -f jacobi
