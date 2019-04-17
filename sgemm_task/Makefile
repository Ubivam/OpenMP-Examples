# C compiler
CC = g++
CC_FLAGS = -fopenmp -O2

prime: 
	$(CC) $(CC_FLAGS) sgemm.cc -o sgemm -lm 
	$(CC)  sgemm_seq.cc -o sgemm_seq -lm
clean:
	rm -f sgemm segmm_seq result*
