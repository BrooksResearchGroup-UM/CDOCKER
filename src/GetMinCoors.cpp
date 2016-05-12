#include "GetMinCoors.h"

void GetMinCoors(int nQuaternions, int nAtoms, float *coors, float* mincoors)
{
  // get the minimum value of x, y and z coordinates of the orientation
  for (int i = 0; i < nQuaternions; i++)
  {
    mincoors[i*3 + 0] = coors[i*nAtoms*3 + 0];
    mincoors[i*3 + 1] = coors[i*nAtoms*3 + 1];
    mincoors[i*3 + 2] = coors[i*nAtoms*3 + 2];
    for (int k = 1; k < nAtoms; k++)
    {
      if (coors[(i*nAtoms+k)*3 + 0] < mincoors[i*3 + 0]) { mincoors[i*3 + 0] = coors[(i*nAtoms+k)*3 + 0]; }
      if (coors[(i*nAtoms+k)*3 + 1] < mincoors[i*3 + 1]) { mincoors[i*3 + 1] = coors[(i*nAtoms+k)*3 + 1]; }
      if (coors[(i*nAtoms+k)*3 + 2] < mincoors[i*3 + 2]) { mincoors[i*3 + 2] = coors[(i*nAtoms+k)*3 + 2]; }
    }
  }
}
