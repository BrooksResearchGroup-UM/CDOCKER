#include "GetMaxCoors.h"

void GetMaxCoors(int nQuaternions, int nAtoms, float *coors, float* maxcoors)
{
  // get the maximum value of x, y and z coordinates of the orientation
  for (int i = 0; i < nQuaternions; i++)
  {
    maxcoors[i*3 + 0] = coors[i*nAtoms*3 + 0];
    maxcoors[i*3 + 1] = coors[i*nAtoms*3 + 1];
    maxcoors[i*3 + 2] = coors[i*nAtoms*3 + 2];
    for (int k = 1; k < nAtoms; k++)
    {
      if (coors[(i*nAtoms+k)*3 + 0] > maxcoors[i*3 + 0]) { maxcoors[i*3 + 0] = coors[(i*nAtoms+k)*3 + 0]; }
      if (coors[(i*nAtoms+k)*3 + 1] > maxcoors[i*3 + 1]) { maxcoors[i*3 + 1] = coors[(i*nAtoms+k)*3 + 1]; }
      if (coors[(i*nAtoms+k)*3 + 2] > maxcoors[i*3 + 2]) { maxcoors[i*3 + 2] = coors[(i*nAtoms+k)*3 + 2]; }
    }
  }
}
