#include <fstream>
#include <iostream>
#include "ReadGrids.h"

void ReadGrids(int &numOfGrids, int &xdim, int &ydim, int &zdim,
	       float &midx, float &midy, float &midz,
	       float &xlen, float &ylen, float &zlen,
	       float &spacing, float &restraint_k,
	       float *&raddi, float* &values,
	       char *fileName)
{
  std::ifstream inFile;
  inFile.open(fileName, std::ios::in);

  if (inFile.fail())
  {
    std::cout << "Unable to open file: " << fileName << std::endl;
    return;
  }
  // ignore 3 line of title
  inFile.ignore(1000,'\n');
  inFile.ignore(1000,'\n');
  inFile.ignore(1000,'\n');

  // read meta parameters for the potential
  inFile >> numOfGrids;
  inFile >> xdim;
  inFile >> ydim;
  inFile >> zdim;
  inFile >> midx;
  inFile >> midy;
  inFile >> midz;
  inFile >> xlen;
  inFile >> ylen;
  inFile >> zlen;
  inFile >> spacing;
  inFile >> restraint_k;

  // read radiis
  int numOfVdwGrids = numOfGrids - 1;
  raddi = new float[numOfVdwGrids];
  for (int i = 0; i < numOfVdwGrids; i++)
  {
    inFile >> raddi[i];
  }
  inFile.ignore(1000,'\n');

  // read values
  int N = xdim * ydim * zdim * numOfGrids;
  values = new float[N];
  float tmp;
  int n = 0;
  for (int i = 0; i < numOfGrids; i++)
  {
    inFile.ignore(1000,'\n');
    for (int j = 0; j < xdim * ydim * zdim; j++)
    {
      inFile >> tmp;
      inFile >> tmp;
      inFile >> tmp;
      inFile >> values[n];
      inFile.ignore(1000,'\n');
      n++;
    }
  }
  inFile.close();
};
