#include <vector>
#include <iostream>
#include <algorithm>

#include "FilterQuaternions.h"

#include "Rotate.h"
#include "GetMinCoors.h"
#include "GetMaxCoors.h"

int FilterQuaternions(float* coor, int nAtom,
		      int nQuater, float* quaternions,
		      float gridLenX, float gridLenY, float gridLenZ,
		      int maxNQuaternionsUsed, float* &quaternionsUsed)
{
  float* coors_all_quaters;
  coors_all_quaters = new float[nQuater*nAtom*3];
  for(int i = 0; i < nQuater; i++)
  {
    for(int j = 0; j < nAtom; j++)
    {
      Rotate(&quaternions[i*4], &coor[j*3], &coors_all_quaters[i*nAtom*3+j*3]);
    }
  }

  float *mincoors_all, *maxcoors_all, *ligandLength_all;
  mincoors_all = new float[nQuater*3]; // minimium coordinates along x, y, and z for all quaternions
  maxcoors_all = new float[nQuater*3]; // maximum coordinates along x, y, and z for all quaternions
  ligandLength_all = new float[nQuater*3]; // lenth along x, y and z for each orientation for all quaternions

  // calculate minimum coor for each quaternions
  GetMinCoors(nQuater, nAtom, coors_all_quaters, mincoors_all);

  // calculate maximum coor for each quaternions
  GetMaxCoors(nQuater, nAtom, coors_all_quaters, maxcoors_all);

  // calculate the length for each quaternion
  for(int i = 0; i < nQuater; i++)
  {
    ligandLength_all[i*3 + 0] = maxcoors_all[i*3 + 0] - mincoors_all[i*3 + 0];
    ligandLength_all[i*3 + 1] = maxcoors_all[i*3 + 1] - mincoors_all[i*3 + 1];
    ligandLength_all[i*3 + 2] = maxcoors_all[i*3 + 2] - mincoors_all[i*3 + 2];
  }

  // index of quaternions which keep the ligand dimenstion smaller than grids
  std::vector <int> idxOfQuatersUsed;
  std::cout << "gridLen: " << gridLenX << "," << gridLenY << "," << gridLenZ << std::endl;
  for(int i = 0; i < nQuater; i++)
  {
    std::cout << i << ":"
	      << ligandLength_all[i*3 + 0] << ","
      	      << ligandLength_all[i*3 + 1] << ","
      	      << ligandLength_all[i*3 + 2] << std::endl;
    if(ligandLength_all[i*3 + 0] < gridLenX && ligandLength_all[i*3 + 1] < gridLenY && ligandLength_all[i*3 + 2] < gridLenZ)
    {
      idxOfQuatersUsed.push_back(i);
    }
  }

  unsigned int numOfQuaternionsUsed = 0;
  if (idxOfQuatersUsed.size() <= maxNQuaternionsUsed)
  {
    numOfQuaternionsUsed = idxOfQuatersUsed.size();
  }
  else
  {
    numOfQuaternionsUsed = maxNQuaternionsUsed;
    std::random_shuffle(idxOfQuatersUsed.begin(), idxOfQuatersUsed.end());
  }
  
  std::cout << "numOfQuaternionsUsed: " << numOfQuaternionsUsed << std::endl;
  delete[] quaternionsUsed;
  quaternionsUsed = new float[numOfQuaternionsUsed*4];
  
  for(int i = 0; i < numOfQuaternionsUsed; i++)
  {
    quaternionsUsed[i*4 + 0] = quaternions[idxOfQuatersUsed[i]*4 + 0];
    quaternionsUsed[i*4 + 1] = quaternions[idxOfQuatersUsed[i]*4 + 1];
    quaternionsUsed[i*4 + 2] = quaternions[idxOfQuatersUsed[i]*4 + 2];
    quaternionsUsed[i*4 + 3] = quaternions[idxOfQuatersUsed[i]*4 + 3];
  }
  delete[] coors_all_quaters;
  delete[] mincoors_all;
  delete[] maxcoors_all;
  delete[] ligandLength_all;
  
  return numOfQuaternionsUsed;
}
