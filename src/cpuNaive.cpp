#include <iostream>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include "time.h"
#include "OpenMM.h"

#include "ReadGrids.h"
#include "ReadQuaternions.h"
#include "Rotate.h"
#include "ReadCrd.h"
#include "GetNonbondedParameters.h"
#include "GetMinCoors.h"
#include "GetIdxOfAtomsForVdwRadius.h"
#include "FillLigandGrid.h"

int main(int argc, char ** argv)
{
  // about ligand
  int nAtoms = atoi(argv[1]);
  float *coor;
  ReadCrd(nAtoms, coor, argv[2]);

  // about grids
  int numOfGrids;
  int xdim, ydim, zdim;
  float midx, midy, midz;
  float xlen, ylen, zlen;
  float spacing, restraint_k;
  float *gridRadii, *gridValues;
  ReadGrids(numOfGrids,  xdim,  ydim,  zdim,
  	    midx, midy, midz,
  	    xlen, ylen, zlen,
  	    spacing, restraint_k,
  	    gridRadii, gridValues,
  	    argv[3]);

  //// read xmlserialized system file and creat OpenMM::System
  std::ifstream sysFile;
  sysFile.open(argv[4], std::ifstream::in);
  if (sysFile.fail())
  {
    std::cout << "Open system file failed: " << argv[4] << std::endl;
    return 1;
  }
  OpenMM::System *sys = new OpenMM::System();
  sys = OpenMM::XmlSerializer::deserialize<OpenMM::System>(sysFile);
  
  ////  get the charge, sigma, epsilon for each particle
  float atomCharges[nAtoms];
  float atomEpsilons[nAtoms];
  float atomRadii[nAtoms];
  GetNonbondedParameters(sys, atomCharges, atomEpsilons, atomRadii);

  //// get which vdw grid will be used and the index of atoms for each vdw grid
  int numOfVdwGrids = numOfGrids - 1;
  int numOfVdwGridsUsed = 0;
  std::vector<int> idxOfVdwUsed;
  std::vector< std::vector<int> > idxOfAtomVdwRadius(numOfVdwGrids);
  GetIdxOfAtomsForVdwRadius(nAtoms, atomRadii, numOfVdwGrids,
  			    gridRadii, numOfVdwGridsUsed,
  			    idxOfVdwUsed, idxOfAtomVdwRadius);
  int numOfGridsUsed = numOfVdwGridsUsed + 1;
  std::cout << "numOfGridsUsed: " << numOfGridsUsed << std::endl;
  
  // copy out the used grids
  float *usedGridValues;
  usedGridValues = new float[(numOfGridsUsed)*xdim*ydim*zdim];
  for(int i = 0; i < idxOfVdwUsed.size(); i++) // this is for vdw grids
  {
    memcpy(&usedGridValues[i*xdim*ydim*zdim],
  	   &gridValues[(idxOfVdwUsed[i])*xdim*ydim*zdim],
  	   xdim*ydim*zdim);
  }
  memcpy(&usedGridValues[numOfVdwGridsUsed*xdim*ydim*zdim],
  	 &gridValues[numOfVdwGrids*xdim*ydim*zdim],
  	 xdim*ydim*zdim); // this is for electrostatic grid
  // clear the memeory for values
  delete[] gridValues;

  // get the min and max for x, y, and z coordinates
  float xmin = coor[0];
  float ymin = coor[1];
  float zmin = coor[2];
  float xmax = xmin;
  float ymax = ymin;
  float zmax = zmin;

  for (int i = 1; i < nAtoms; i++)
  {
    if (coor[3*i + 0] < xmin) xmin = coor[3*i + 0];
    if (coor[3*i + 1] < ymin) ymin = coor[3*i + 1];
    if (coor[3*i + 2] < zmin) zmin = coor[3*i + 2];
    if (coor[3*i + 0] > xmax) xmax = coor[3*i + 0];
    if (coor[3*i + 1] > ymax) ymax = coor[3*i + 1];
    if (coor[3*i + 2] > zmax) zmax = coor[3*i + 2];
  }

  float xlen_l = xmax - xmin;
  float ylen_l = ymax - ymin;
  float zlen_l = zmax - zmin;

  int xdim_l = int(xlen_l / spacing) + 2;
  int ydim_l = int(ylen_l / spacing) + 2;
  int zdim_l = int(zlen_l / spacing) + 2;
  int n_l = xdim_l * ydim_l * zdim_l;
  
  std::cout << "xdim_l: " << xdim_l << std::endl;
  std::cout << "ydim_l: " << ydim_l << std::endl;
  std::cout << "zdim_l: " << zdim_l << std::endl;
  
  float ligandGridValues[numOfGridsUsed * xdim_l*ydim_l*zdim_l];
  memset(ligandGridValues, 0, numOfGridsUsed * xdim_l*ydim_l*zdim_l * sizeof(float));  
  float x,y,z;
  int xIndex, yIndex, zIndex, index;
  float xRatio, yRatio, zRatio;  
  for (int i = 0; i < idxOfVdwUsed.size(); i++)
  {
    for (int j = 0; j < idxOfAtomVdwRadius[idxOfVdwUsed[i]].size(); j++)
    {
      int m = idxOfAtomVdwRadius[idxOfVdwUsed[i]][j];
      x = coor[m*3 + 0];
      y = coor[m*3 + 1];
      z = coor[m*3 + 2];
      xIndex = int((x - xmin)/spacing);
      yIndex = int((y - ymin)/spacing);
      zIndex = int((z - zmin)/spacing);
      xRatio = ((x - xmin) - xIndex * spacing)/spacing;
      yRatio = ((y - ymin) - yIndex * spacing)/spacing;
      zRatio = ((z - zmin) - zIndex * spacing)/spacing;
      assert(xIndex < xdim_l);
      assert(yIndex < ydim_l);
      assert(zIndex < zdim_l);
      index = i * n_l + (xIndex * ydim_l + yIndex) * zdim_l + zIndex;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * (1 - yRatio) * (1 - zRatio);
      index = i * n_l + (xIndex * ydim_l + yIndex) * zdim_l + zIndex + 1;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * (1 - yRatio) * zRatio;
      index = i * n_l + (xIndex * ydim_l + yIndex + 1) * zdim_l + zIndex;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * yRatio * (1 - zRatio);
      index = i * n_l + (xIndex * ydim_l + yIndex + 1) * zdim_l + zIndex + 1;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * (1 - xRatio) * yRatio * zRatio;
      index = i * n_l + ((xIndex + 1) * ydim_l + yIndex) * zdim_l + zIndex;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * xRatio * (1 - yRatio) * (1 - zRatio);
      index = i * n_l + ((xIndex + 1) * ydim_l + yIndex) * zdim_l + zIndex + 1;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * xRatio * (1 - yRatio) * zRatio;
      index = i * n_l + ((xIndex + 1) * ydim_l + yIndex + 1) * zdim_l + zIndex;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * xRatio * yRatio * (1 - zRatio);
      index = i * n_l + ((xIndex + 1) * ydim_l + yIndex + 1) * zdim_l + zIndex + 1;
      ligandGridValues[index] += sqrt(atomEpsilons[m]) * xRatio * yRatio * zRatio;
    }
  }

  for (int m = 0; m < nAtoms; m++)
  {
    x = coor[m*3 + 0];
    y = coor[m*3 + 1];
    z = coor[m*3 + 2];
    xIndex = int((x - xmin)/spacing);
    yIndex = int((y - ymin)/spacing);
    zIndex = int((z - zmin)/spacing);
    xRatio = ((x - xmin) - xIndex * spacing)/spacing;
    yRatio = ((y - ymin) - yIndex * spacing)/spacing;
    zRatio = ((z - zmin) - zIndex * spacing)/spacing;

    index = numOfVdwGridsUsed * n_l + (xIndex * ydim_l + yIndex) * zdim_l + zIndex;
    ligandGridValues[index] += atomCharges[m] * (1 - xRatio) * (1 - yRatio) * (1 - zRatio);
    index = numOfVdwGridsUsed * n_l + (xIndex * ydim_l + yIndex) * zdim_l + zIndex + 1;
    ligandGridValues[index] += atomCharges[m] * (1 - xRatio) * (1 - yRatio) * zRatio;
    index = numOfVdwGridsUsed * n_l + (xIndex * ydim_l + yIndex + 1) * zdim_l + zIndex;
    ligandGridValues[index] += atomCharges[m] * (1 - xRatio) * yRatio * (1 - zRatio);
    index = numOfVdwGridsUsed * n_l + (xIndex * ydim_l + yIndex + 1) * zdim_l + zIndex + 1;
    ligandGridValues[index] += atomCharges[m] * (1 - xRatio) * yRatio * zRatio;
    index = numOfVdwGridsUsed * n_l + ((xIndex + 1) * ydim_l + yIndex) * zdim_l + zIndex;
    ligandGridValues[index] += atomCharges[m] * xRatio * (1 - yRatio) * (1 - zRatio);
    index = numOfVdwGridsUsed * n_l + ((xIndex + 1) * ydim_l + yIndex) * zdim_l + zIndex + 1;
    ligandGridValues[index] += atomCharges[m] * xRatio * (1 - yRatio) * zRatio;
    index = numOfVdwGridsUsed * n_l + ((xIndex + 1) * ydim_l + yIndex + 1) * zdim_l + zIndex;
    ligandGridValues[index] += atomCharges[m] * xRatio * yRatio * (1 - zRatio);
    index = numOfVdwGridsUsed * n_l + ((xIndex + 1) * ydim_l + yIndex + 1) * zdim_l + zIndex + 1;
    ligandGridValues[index] += atomCharges[m] * xRatio * yRatio * zRatio;
  }
  
  // calculate energy
  int xdim_off = xdim - xdim_l;
  int ydim_off = ydim - ydim_l;
  int zdim_off = zdim - zdim_l;

  float energy[xdim_off*ydim_off*zdim_off];
  memset(energy, 0, xdim_off*ydim_off*zdim_off*sizeof(float));

  int tmp;
  clock_t t;
  t = clock();
  for(int i = 0; i < xdim_off; i++)
  {
    for(int j = 0; j < ydim_off; j++)
    {
      for(int k = 0; k < zdim_off; k++)
      {
  	tmp = (i * ydim_off + j) * zdim_off + k;
  	for (int l = 0; l < numOfVdwGridsUsed + 1; l++)
  	{
  	  for (int m = 0; m < xdim_l; m++)
  	  {
  	    for (int n = 0; n < ydim_l; n++)
  	    {
  	      for (int o = 0; o < zdim_l; o++)
  	      {
		energy[tmp] = energy[tmp] + (ligandGridValues[l*n_l + (m * ydim_l + n) * zdim_l + o] * usedGridValues[l*xdim*ydim*zdim + ((m + i)*ydim + (n + j)) * zdim + o + k]);
  	      }
  	    }
  	  }
  	}
      }
    }
  }

  t = clock() - t;
  std::cout << "Energy: " << energy[0] << std::endl;
  std::cout << "Energy: " << energy[1] << std::endl;
  std::cout << "Number of clicks: " << t << std::endl;
  std::cout << "CPU time: " << ((float)t)/CLOCKS_PER_SEC << " seconds" << std::endl;
  
  return 0;
}
