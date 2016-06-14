#include <random>
#include <math.h>
#include "QuaternionUniformSampling.h"

void QuaternionUniformSampling(std::mt19937_64& gen, float* &quaters, int nQuaters)
{
  std::uniform_real_distribution<double> udist(0,1);
  delete[] quaters;
  quaters = new float[4*nQuaters];
  double s, sigma1, sigma2, theta1, theta2;
  for(int i = 0; i < nQuaters; i++)
  {
    s = udist(gen);
    sigma1 = sqrt(1-s);
    sigma2 = sqrt(s);
    theta1 = 2*M_PI*udist(gen);
    theta2 = 2*M_PI*udist(gen);
    
    quaters[i*4 + 0] = cos(theta2)*sigma2;
    quaters[i*4 + 1] = sin(theta1)*sigma1;
    quaters[i*4 + 2] = cos(theta1)*sigma1;
    quaters[i*4 + 3] = sin(theta2)*sigma2;
  }
}
