#include <fstream>
#include <iostream>
#include "ReadQuaternions.h"

void ReadQuaternions(int R, float *&quaternions, char* fileName)
{
  std::ifstream inFile;
  inFile.open(fileName, std::ifstream::in);
  quaternions = new float[R*4];
  
  if (inFile.fail())
  {
    std::cout << "Open quaternioins file failed!" << std::endl;
    return;
  }

  for (int i = 0; i < R; i++)
  {
    inFile >> quaternions[4*i];
    inFile >> quaternions[4*i+1];
    inFile >> quaternions[4*i+2];
    inFile >> quaternions[4*i+3];
  }
  inFile.close();
};
