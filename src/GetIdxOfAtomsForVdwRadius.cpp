#include <cmath>
#include "GetIdxOfAtomsForVdwRadius.h"

void GetIdxOfAtomsForVdwRadius(int nAtoms, float* atomRadii,
			       int numOfVdwGrids, float*gridRadii,
			       int &numOfVdwGridsUsed, std::vector<int> &idxOfVdwUsed,
			       std::vector< std::vector<int> > &idxOfAtomVdwRadius)
{
  // get the idx of atoms for each vdw radius
  for(int i = 0; i < nAtoms; i++)
  {
    int idx = 0;
    double diff = std::abs(atomRadii[i] - gridRadii[idx]);
    for (int j = 1; j < numOfVdwGrids; j++)
    {
      if (std::abs(atomRadii[i] - gridRadii[j]) < diff)
      {
	idx = j;
	diff = std::abs(atomRadii[i] - gridRadii[j]);
      }
    }
    idxOfAtomVdwRadius[idx].push_back(i);
  }

  // num of vdw radii used and its index
  numOfVdwGridsUsed = 0;
  for(int i = 0; i < numOfVdwGrids; i++)
  {
    if (idxOfAtomVdwRadius[i].size() > 0)
    {
      numOfVdwGridsUsed += 1;
      idxOfVdwUsed.push_back(i);
    }
  }
}
