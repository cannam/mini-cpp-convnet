
CXXFLAGS	:= -std=c++14 -O3 -Wall -Wextra -Wno-unused-parameter -I../ext/tiny-dnn

LDFLAGS		:= -lpng -lpthread

with-tiny-dnn:	with-tiny-dnn.cpp ../weights.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

../weights.o:	../weights.cpp

