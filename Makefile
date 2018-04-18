
CXXFLAGS	:= -std=c++11 -O3 -Wall -Wextra -Wno-unused-parameter

LDFLAGS		:= -lpng

flower:	flower.cpp weights.o
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

weights.o:	weights.cpp

clean:
	rm *.o
