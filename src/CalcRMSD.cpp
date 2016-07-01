#include "CalcRMSD.h"

#include <math.h>
#include <vector>
#include <assert.h>
#include <openbabel/mol.h>

// calcuate the RMSD between two OBMols without doing alignment and not including Hydrogen
double CalcRMSD(OpenBabel::OBMol &mol1, OpenBabel::OBMol &mol2)
{
  std::vector<double> coor1;
  std::vector<double> coor2;
  int n1 = mol1.NumAtoms();
  int n2 = mol2.NumAtoms();
  int numOfH = 0;
  assert(n1 == n2);

  // get the coordinates of heavy atoms
  OpenBabel::OBAtom *a;
  for (int i = 0; i < n1; i++)
  {
    a = mol1.GetAtomById(i);
    if (a->GetAtomicNum() == 1) { break; }
    numOfH += 1;
    coor1.push_back(a->GetVector()[0]);
    coor1.push_back(a->GetVector()[1]);
    coor1.push_back(a->GetVector()[2]);

    a = mol2.GetAtomById(i);
    coor2.push_back(a->GetVector()[0]);
    coor2.push_back(a->GetVector()[1]);
    coor2.push_back(a->GetVector()[2]);
  }

  // calcualte the rmsd
  double totalSquareDeviation = 0;
  for (int i = 0; i < numOfH * 3; i++)
  {
    totalSquareDeviation += (coor1[i] - coor2[i]) * (coor1[i] - coor2[i]);
  }
  
  return sqrt(totalSquareDeviation/numOfH);
}
