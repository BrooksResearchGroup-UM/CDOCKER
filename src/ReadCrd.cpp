#include <fstream>
#include <iostream>
#include <string>
#include "ReadCrd.h"

void ReadCrd(int nAtoms, float *&coor, std::string fileName)
{
  std::ifstream coorFile;
  coorFile.open(fileName, std::ifstream::in);
  if (coorFile.fail())
  {
    std::cout << "Open coor file failed!" << std::endl;
    return;
  }
    
  // ignore the title part
  bool isAtom = false;
  std::string line;
  while (!isAtom)
  {
    std::getline(coorFile, line);    
    if (line[0] != '*')
    {
      isAtom = true;
    }
  }
  
  // ignore the atom index, atom type and structure name in each row
  int junkInt;
  std::string junkString;
  coor = new float[nAtoms * 3];
    
  for (int i = 0; i < nAtoms; i++)
  {
    coorFile >> junkInt;
    coorFile >> junkInt;
    coorFile >> junkString;
    coorFile >> junkString;
    coorFile >> coor[3*i]; // this is the x coordinate of atom i
    coorFile >> coor[3*i+1]; // this is the y coordinate of atom i
    coorFile >> coor[3*i+2]; // this is the z coordinate of atom i
    coorFile.ignore(1000, '\n');
  }
  coorFile.close();
};
