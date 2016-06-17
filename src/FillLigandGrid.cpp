#include <vector>
#include <math.h>
#include <iostream>

#include "FillLigandGrid.h"

void FillLigandGrid(int nQuaternions,
		    int nAtoms, float *coors, float *mincoors,
		    float *atomCharges, float *atomEpsilons,
		    int numOfVdwGridsUsed, std::vector<int> idxOfVdwUsed,
		    std::vector< std::vector<int> > &idxOfAtomVdwRadius,
		    int xdim, int ydim, int zdim,
		    float spacing, float *ligandGridValues
		    )
{
  // fill the vdw grids
  for (int i = 0; i < nQuaternions; i++)
  {
    for (int j = 0; j < idxOfVdwUsed.size(); j++)
    {
      for (int l = 0; l < idxOfAtomVdwRadius[idxOfVdwUsed[j]].size(); l++)
      {	
	int m = idxOfAtomVdwRadius[idxOfVdwUsed[j]][l];
	double x = coors[(i*nAtoms + m)*3 + 0];
	double y = coors[(i*nAtoms + m)*3 + 1];
	double z = coors[(i*nAtoms + m)*3 + 2];
	// index of grids this atom sits
	int xIndex = int((x - mincoors[i*3 + 0])/spacing);
	int yIndex = int((y - mincoors[i*3 + 1])/spacing);
	int zIndex = int((z - mincoors[i*3 + 2])/spacing);
	
	// trilinear interpolation
	double xRatio = ((x - mincoors[i*3 + 0]) - xIndex * spacing)/spacing;
	double yRatio = ((y - mincoors[i*3 + 1]) - yIndex * spacing)/spacing;
	double zRatio = ((z - mincoors[i*3 + 2]) - zIndex * spacing)/spacing;
	
	// index of ligandGridValue
	int idxGrid = i * (numOfVdwGridsUsed + 1) + j;
	int idxValue = idxGrid * xdim * ydim * zdim;

	int index = (xIndex * ydim + yIndex) * zdim + zIndex;
	
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * (1 - yRatio) * (1 - zRatio);
	
	// std::cout << "Here we go 2" << std::endl;
	
	index = (xIndex * ydim + yIndex) * zdim + zIndex + 1;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * (1 - yRatio) * zRatio;
	
	index = (xIndex * ydim + yIndex + 1) * zdim + zIndex;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * yRatio * (1 - zRatio);
	
	index = (xIndex * ydim + yIndex + 1) * zdim + zIndex + 1;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * yRatio * zRatio;
	
	index = ((xIndex + 1) * ydim + yIndex) * zdim + zIndex;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * xRatio * (1 - yRatio) * (1 - zRatio);
	
	index = ((xIndex + 1) * ydim + yIndex) * zdim + zIndex + 1;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * xRatio * (1 - yRatio) * zRatio;
	
	index = ((xIndex + 1) * ydim + yIndex + 1) * zdim + zIndex;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * xRatio * yRatio * (1 - zRatio);
	
	index = ((xIndex + 1) * ydim + yIndex + 1) * zdim + zIndex + 1;
	ligandGridValues[idxValue + index] += sqrt(atomEpsilons[m]) * xRatio * yRatio * zRatio;

	

      }
    }
  }

  // fill the electricstatic grid
  for (int i = 0; i < nQuaternions; i++)
  {
    for (int m = 0; m < nAtoms; m++)
    {
      double x = coors[(i*nAtoms + m)*3 + 0];
      double y = coors[(i*nAtoms + m)*3 + 1];
      double z = coors[(i*nAtoms + m)*3 + 2];
      // index of grids this atom sits
      int xIndex = int((x - mincoors[i*3 + 0])/spacing);
      int yIndex = int((y - mincoors[i*3 + 1])/spacing);
      int zIndex = int((z - mincoors[i*3 + 2])/spacing);
	
      // trilinear interpolation
      double xRatio = ((x - mincoors[i*3 + 0]) - xIndex * spacing)/spacing;
      double yRatio = ((y - mincoors[i*3 + 1]) - yIndex * spacing)/spacing;
      double zRatio = ((z - mincoors[i*3 + 2]) - zIndex * spacing)/spacing;

      int idxGrid = i * (numOfVdwGridsUsed + 1) + numOfVdwGridsUsed;
      int idxValue = idxGrid * xdim * ydim * zdim;
	
      int index = (xIndex * ydim + yIndex) * zdim + zIndex;
      ligandGridValues[idxValue + index] += atomCharges[m] * (1 - xRatio) * (1 - yRatio) * (1 - zRatio);
	
      index = (xIndex * ydim + yIndex) * zdim + zIndex + 1;
      ligandGridValues[idxValue + index] += atomCharges[m] * (1 - xRatio) * (1 - yRatio) * zRatio;
	
      index = (xIndex * ydim + yIndex + 1) * zdim + zIndex;
      ligandGridValues[idxValue + index] += atomCharges[m] * (1 - xRatio) * yRatio * (1 - zRatio);
	
      index = (xIndex * ydim + yIndex + 1) * zdim + zIndex + 1;
      ligandGridValues[idxValue + index] += atomCharges[m] * (1 - xRatio) * yRatio * zRatio;
	
      index = ((xIndex + 1) * ydim + yIndex) * zdim + zIndex;
      ligandGridValues[idxValue + index] += atomCharges[m] * xRatio * (1 - yRatio) * (1 - zRatio);
	
      index = ((xIndex + 1) * ydim + yIndex) * zdim + zIndex + 1;
      ligandGridValues[idxValue + index] += atomCharges[m] * xRatio * (1 - yRatio) * zRatio;
	
      index = ((xIndex + 1) * ydim + yIndex + 1) * zdim + zIndex;
      ligandGridValues[idxValue + index] += atomCharges[m] * xRatio * yRatio * (1 - zRatio);
	
      index = ((xIndex + 1) * ydim + yIndex + 1) * zdim + zIndex + 1;
      ligandGridValues[idxValue + index] += atomCharges[m] * xRatio * yRatio * zRatio;
    }
  }
}
