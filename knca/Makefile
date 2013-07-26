## Copyright 2011 and onwards
## Author: dtarlow@cs.toronto.edu (Danny Tarlow)
## 12/01/2011
##
## Makefile for graph cuts neural network component

CC = g++
LD = g++

## external .h files to bring in
EXTRA_INCLUDES =

# Danny's laptop
#CCFLAGS = -g -Wall -Wno-deprecated -fPIC
#LDFLAGS = 
#DYLIB_CMD = -dynamiclib -fPIC

# Toronto clusters
#CCFLAGS = -g -Wall -Wno-deprecated -I/u/dtarlow/lib/include/ -I/u/dtarlow/workspace/lib/include/ -fPIC
#LDFLAGS = -lm -L/u/dtarlow/lib/lib/ -L/u/dtarlow/workspace/lib/lib/ -lboost_program_options

CCFLAGS = -g -Wall -Wno-deprecated -fPIC
LDFLAGS = -lm
DYLIB_CMD = -fPIC -shared

## sources
PROGFILE = knca_alg.cpp 
CCFILES = 

## object compilation
PROGOBJS = $(CCFILES:.cpp=.o) $(PROGFILE:.cpp=.o)

# lib compilation
CHAINLIB = knca_alg.A.dylib
#CARDLIB = cardprop.A.dylib

## binary compilation
PROG = knca

all: 
	@echo "-> Compiling utils and external sources..."	
	@make knca_alg.o
	@echo "-> Building $(PROG) executable..."
	@make $(PROG) 
	@echo "-> Building $(CHAINLIB) lib..."
	@make $(CHAINLIB)
	@echo

knca_alg.o: knca_alg.cpp
	$(CC) $(CCFLAGS) $(EXTRA_INCLUDES) -c knca_alg.cpp

$(PROG): $(PROGOBJS)
	$(LD) $(LDFLAGS) $(PROGOBJS) -o $(PROG) $(CCFLAGS) $(LDFLAGS)  $(CARBON_FLAGS)

$(CHAINLIB): $(PROGOBJS)
	$(LD) $(DYLIB_CMD) $(PROGOBJS) $(LDFLAGS) -o $(CHAINLIB)

clean:
	@echo "-> Cleaning..."
	rm -f .cpp.o $(PROG) $(PROGOBJS) $(GCLIB)
