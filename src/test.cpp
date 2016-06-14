#include <iostream>

#include "QuaternionUniformSampling.h"


int main(int argc, char** argv)
{
  int N = 1000;
  float* quaters;
  
  std::random_device rd;
  std::mt19937_64 gen(rd());
  QuaternionUniformSampling(gen, quaters, N);

  for(int i = 0; i < N; i++)
  {
    std::cout << quaters[i*4 + 0] << ","
	      << quaters[i*4 + 1] << ","
	      << quaters[i*4 + 2] << ","
	      << quaters[i*4 + 3] << std::endl;
  }
  return 0;
}
