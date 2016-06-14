#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>

#include "OpenMM.h"
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <cufft.h>
#include "ReadCrd.h"
#include "ReadGrids.h"
#include "ReadQuaternions.h"
#include "Rotate.h"
#include "GetNonbondedParameters.h"
#include "GetMinCoors.h"
#include "GetMaxCoors.h"
#include "GetIdxOfAtomsForVdwRadius.h"
#include "FillLigandGrid.h"
#include "GeneConformations.h"
#include "kernel.h"
#include "QuaternionUniformSampling.h"
#include "FilterQuaternions.h"

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

//// main function ////
int main(int argc, char** argv)
{
  OpenMM::Platform::loadPluginsFromDirectory(
  					     "/home/xqding/apps/openmmDev/lib/plugins");
  
  // read molecule
  std::string fileName(argv[1]);
  OpenBabel::OBMol mol;
  OpenBabel::OBConversion conv(&std::cin, &std::cout);
  conv.SetInFormat("mol2");
  conv.SetOutFormat("pdb");
  conv.ReadFile(&mol, fileName);
  int nAtom = mol.NumAtoms();
  
  // read system
  std::ifstream sysFile;
  sysFile.open(argv[2], std::ifstream::in);
  if (sysFile.fail())
  {
    std::cout << "Open system file failed: " << argv[2] << std::endl;    return 1;
  }
  OpenMM::System *sys = new OpenMM::System();
  sys = OpenMM::XmlSerializer::deserialize<OpenMM::System>(sysFile);
  
  // get nonbonded parameters
  float atomCharges[nAtom];
  float atomEpsilons[nAtom];
  float atomRadii[nAtom];
  GetNonbondedParameters(sys, atomCharges, atomEpsilons, atomRadii);

  // read grid potential
  int numOfGrids, xdim, ydim, zdim;
  float midx, midy, midz;
  float xlen, ylen, zlen;
  float spacing, restraint_k;
  float *gridRadii, *gridValues;
  ReadGrids(numOfGrids, xdim, ydim, zdim,
  	    midx, midy, midz,
  	    xlen, ylen, zlen,
  	    spacing, restraint_k,
  	    gridRadii, gridValues,
  	    argv[3]);
  int numOfVdwGrids = numOfGrids - 1;
  
  // add electrostatic grid force
  double xmin = midx - xlen / 2;
  double ymin = midy - ylen / 2;
  double zmin = midz - zlen / 2;
  double xmax = xmin + (xdim - 1) * spacing;
  double ymax = ymin + (ydim - 1) * spacing;
  double zmax = zmin + (zdim - 1) * spacing;

  // get index of atoms for each vdw radius
  int numOfVdwGridsUsed;
  std::vector<int> idxOfVdwUsed;
  std::vector< std::vector<int> > idxOfAtomVdwRadius(numOfVdwGrids);
  GetIdxOfAtomsForVdwRadius(nAtom, atomRadii,
  			    numOfVdwGrids, gridRadii,
  			    numOfVdwGridsUsed, idxOfVdwUsed,
  			    idxOfAtomVdwRadius);
  int numOfGridsUsed = numOfVdwGridsUsed + 1;

  // copy out the potential grids which are used
  float *usedGridValues;
  usedGridValues = new float[numOfGridsUsed*xdim*ydim*zdim];
  for(int i = 0; i < numOfVdwGridsUsed; i++)
  {
    memcpy(&usedGridValues[i*xdim*ydim*zdim],
  	   &gridValues[idxOfVdwUsed[i]*xdim*ydim*zdim],
  	   sizeof(float)*xdim*ydim*zdim);
  }
  memcpy(&usedGridValues[numOfVdwGridsUsed*xdim*ydim*zdim],
  	 &gridValues[numOfVdwGrids*xdim*ydim*zdim],
  	 sizeof(float)*xdim*ydim*zdim);

  //// do translation and rotation search using FFT
  // batch cudaFFT for potential grids
  int n[3];
  n[0] = xdim;
  n[1] = ydim;
  n[2] = zdim;
  int inembed[3];
  inembed[0] = xdim;
  inembed[1] = ydim;
  inembed[2] = zdim;
  int idist = inembed[0] * inembed[1] * inembed[2];
  int istride = 1;
  
  int onembed[3];
  onembed[0] = xdim;
  onembed[1] = ydim;
  onembed[2] = zdim/2 + 1;
  int odist = onembed[0] * onembed[1] * onembed[2];
  int ostride = 1;
  int nBatchPotential = numOfGridsUsed;
  
  cufftReal* d_potential_f;
  cudaMalloc((void **)&d_potential_f, sizeof(cufftReal)*nBatchPotential*idist);
  cudaMemcpy(d_potential_f, usedGridValues,
  	     sizeof(cufftReal)*nBatchPotential*idist,
  	     cudaMemcpyHostToDevice);
  cufftComplex *d_potential_F;
  cudaMalloc((void **)&d_potential_F, sizeof(cufftComplex)*nBatchPotential*odist);
  cufftHandle potentialPlan;
  cufftResult potentialRes = cufftPlanMany(&potentialPlan, 3, n,
  					   inembed, istride, idist,
  					   onembed, ostride, odist,
  					   CUFFT_R2C, nBatchPotential);
  if (potentialRes != CUFFT_SUCCESS)
  {
    std::cout << "Potential plan creat failed!";
    return 1;
  }
  potentialRes = cufftExecR2C(potentialPlan, d_potential_f, d_potential_F);
  if (potentialRes != CUFFT_SUCCESS)
  {
    std::cout << "Potential transform failed!";
    return 1;
  }

  // // read quaternions
  // int numOfTotalQuaternions = atoi(argv[4]);
  // std::cout << "numOfTotalQuaternions: " << numOfTotalQuaternions << std::endl;
  // float *quaternions;
  // quaternions = new float[numOfTotalQuaternions * 4];
  // ReadQuaternions(numOfTotalQuaternions, quaternions, argv[5]);

  // generate uniform quaternions
  int numOfTotalQuaternions = atoi(argv[4]);
  float* quaternions;
  std::random_device rd;
  std::mt19937_64 gen(rd());
  QuaternionUniformSampling(gen, quaternions, numOfTotalQuaternions);

  // get coor
  float* coor;
  coor = new float[nAtom*3];
  for(int i = 0; i < nAtom; i++)
  {
    coor[i*3 + 0] = (float) mol.GetCoordinates()[i*3 + 0];
    coor[i*3 + 1] = (float) mol.GetCoordinates()[i*3 + 1];
    coor[i*3 + 2] = (float) mol.GetCoordinates()[i*3 + 2];
  }

  // ignore quaterions, whose end structures' dimenstion is larger than the grids
  size_t maxNQuaternionsUsed = atoi(argv[5]);
  size_t numOfQuaternionsUsed;
  float* quaternionsUsed = 0;  
  numOfQuaternionsUsed = FilterQuaternions(coor, nAtom,
  					   numOfTotalQuaternions, quaternions,
  					   xlen, ylen, zlen,
  					   maxNQuaternionsUsed,quaternionsUsed);
  
  // loop over all batches for different orientation
  int numOfQuaternionsOneBatch = 100;
  int numOfBatches = numOfQuaternionsUsed / numOfQuaternionsOneBatch + 1;
    
  // allocate the data structures which will be used
  float *coors; // rotated coordinates
  coors = new float[numOfQuaternionsOneBatch*nAtom*3];
  float mincoors[numOfQuaternionsOneBatch*3]; // minimium coordinates along x, y, and z
  float maxcoors[numOfQuaternionsOneBatch*3]; // maximum coordinates along x, y, and z
  float ligandLength[numOfQuaternionsOneBatch*3]; // lenth along x, y and z for each orientation
  
  float *ligandGridValues; // grid for ligand
  ligandGridValues = new float[numOfQuaternionsOneBatch*numOfGridsUsed*xdim*ydim*zdim];
  
  // cudaFFT for ligand grid
  int nBatchLigand = numOfQuaternionsOneBatch*numOfGridsUsed;
  cufftReal* d_ligand_f;
  cudaMalloc((void **)&d_ligand_f, sizeof(cufftReal)*nBatchLigand*idist);
  cufftComplex * d_ligand_F;
  cudaMalloc((void **)&d_ligand_F, sizeof(cufftComplex)*nBatchLigand*odist);
  cufftHandle ligandPlan;
  cufftResult ligandRes = cufftPlanMany(&ligandPlan, 3, n,
  					inembed, istride, idist,
  					onembed, ostride, odist,
  					CUFFT_R2C, nBatchLigand);
  if (ligandRes != CUFFT_SUCCESS)
  {
    std::cout << "ligand plan creat failed!";
    return 1;
  }

  dim3 threads_ConjMult(1024, 1, 1);
  dim3 blocks_ConjMult((numOfQuaternionsOneBatch*numOfGridsUsed*odist)/(1024*1024) + 1,1024,1);
 
  cufftComplex * d_ligand_sum_F;
  cudaMalloc((void **)&d_ligand_sum_F, sizeof(cufftComplex)*numOfQuaternionsOneBatch*odist);
  
  dim3 threads_SumGrids(1024, 1, 1);
  dim3 blocks_SumGrids((numOfQuaternionsOneBatch*odist)/(1024*1024) + 1,1024,1);
  
  cufftReal *d_ligand_sum_f;
  cudaMalloc((void **)&d_ligand_sum_f, sizeof(cufftReal)*numOfQuaternionsOneBatch*idist);
  cufftHandle ligandRPlan;
  cufftResult ligandRRes = cufftPlanMany(&ligandRPlan, 3, n,
  					 onembed, ostride, odist,
  					 inembed, istride, idist,
  					 CUFFT_C2R, numOfQuaternionsOneBatch);
  if (ligandRRes != CUFFT_SUCCESS)
  {
    std::cout << "ligand reverse plan creat failed!" << std::endl;
    std::cout << "Error code: " << ligandRRes << std::endl;
    return 1;
  }
  float* energy;
  energy = new float[numOfQuaternionsOneBatch*idist];

  // save information for each quaternion
  int minEnergyIdxX[numOfQuaternionsUsed];
  int minEnergyIdxY[numOfQuaternionsUsed];
  int minEnergyIdxZ[numOfQuaternionsUsed];
  std::vector <float> minEnergyAllQ(numOfQuaternionsUsed);
  for(int i = 0; i < numOfQuaternionsUsed; i++)
  {
    minEnergyAllQ[i] = INFINITY;
  }

  // do the calculation
  for(int idxOfBatch = 0; idxOfBatch < numOfBatches; idxOfBatch++)
  {
    std::cout << "idxOfBatch: " << idxOfBatch << std::endl;
    // rotate
    for(int i = 0; i < numOfQuaternionsOneBatch; i++)
    {
      if (idxOfBatch*numOfQuaternionsOneBatch + i < numOfQuaternionsUsed)
      {
	for(int j = 0; j < nAtom; j++)
	{
	  Rotate(&quaternionsUsed[(idxOfBatch*numOfQuaternionsOneBatch + i)*4], &coor[j*3], &coors[i*nAtom*3+j*3]);
	}
      }
    }
    
    // calculate minimum coor for each quaternions
    GetMinCoors(numOfQuaternionsOneBatch, nAtom, coors, mincoors);

    // calculate maximum coor for each quaternions
    GetMaxCoors(numOfQuaternionsOneBatch, nAtom, coors, maxcoors);

    // calculate the length for each quaternion
    for(int i = 0; i < numOfQuaternionsOneBatch; i++)
    {
      ligandLength[i*3 + 0] = maxcoors[i*3 + 0] - mincoors[i*3 + 0];
      ligandLength[i*3 + 1] = maxcoors[i*3 + 1] - mincoors[i*3 + 1];
      ligandLength[i*3 + 2] = maxcoors[i*3 + 2] - mincoors[i*3 + 2];
    }
      
    // fill ligand grid
    memset(ligandGridValues, 0, sizeof(float)*numOfQuaternionsOneBatch*numOfGridsUsed*xdim*ydim*zdim);
    FillLigandGrid(numOfQuaternionsOneBatch,
		   nAtom, coors, mincoors,
		   atomCharges, atomEpsilons,
		   numOfVdwGridsUsed, idxOfVdwUsed,
		   idxOfAtomVdwRadius,
		   xdim, ydim, zdim,
		   spacing, ligandGridValues);


    // batch cudaFFT for ligand grid
    cudaMemcpy(d_ligand_f, ligandGridValues,
	       sizeof(cufftReal)*nBatchLigand*idist,
	       cudaMemcpyHostToDevice);
    ligandRes = cufftExecR2C(ligandPlan, d_ligand_f, d_ligand_F);
    if (ligandRes != CUFFT_SUCCESS)
    {
      std::cout << "ligand transform failed!";
      return 1;
    }

    // calcualte energy using reverse FFT
    ConjMult <<<blocks_ConjMult, threads_ConjMult>>> (d_potential_F, d_ligand_F, odist, numOfGridsUsed);
    CUDA_CHECK();

    SumGrids <<<blocks_SumGrids, threads_SumGrids>>> (d_ligand_F, d_ligand_sum_F, numOfGridsUsed, odist, idist);
    CUDA_CHECK();

    ligandRRes = cufftExecC2R(ligandRPlan, d_ligand_sum_F, d_ligand_sum_f);
    if (ligandRRes != CUFFT_SUCCESS)
    {
      std::cout << "ligand reverse transform failed!";
      return 1;
    }

    // copy energy back
    cudaMemcpy(energy, d_ligand_sum_f, sizeof(float)*numOfQuaternionsOneBatch*idist,
	       cudaMemcpyDeviceToHost);

    // record the minimum energy pose in terms of quaternions, x, y and z
    // this needs to be rewriten as a GPU kernel in the future
    for(int q = 0; q < numOfQuaternionsOneBatch; q++)
    {
      int idxQ = idxOfBatch * numOfQuaternionsOneBatch + q;
      for(int i = 0; i < (xdim-int(ligandLength[q*3+0]/spacing)-2); i++)
      {
	for(int j = 0; j < (ydim-int(ligandLength[q*3+1]/spacing)-2); j++)
	{
	  for(int k = 0; k < (zdim-int(ligandLength[q*3+2]/spacing)-2); k++)
	  {
	    if(idxOfBatch*numOfQuaternionsOneBatch + q < numOfQuaternionsUsed)
	    {
	      int tmp = q*idist + (i*ydim + j)*zdim + k;
	      if((energy[tmp]/sqrt(idist)) < minEnergyAllQ[idxQ])
	      {
		minEnergyAllQ[idxQ] = energy[tmp] / sqrt(idist);
		minEnergyIdxX[idxQ] = i;
		minEnergyIdxY[idxQ] = j;
		minEnergyIdxZ[idxQ] = k;
	      }
	    }
	  }
	}
      }
    }
    
  }

  // calculate the coordinates corresponding to lowest nLowest energy orientation
  std::vector<size_t> idxLowestQ;
  idxLowestQ = sort_index<float>(minEnergyAllQ);

  int nLowest = atoi(argv[6]);
  for(int iLowest = 0; iLowest < nLowest && iLowest < numOfQuaternionsUsed; iLowest++)
  {
    float minEnergyCoor[nAtom*3];
    for(int i = 0; i < nAtom; i++)
    {
      Rotate(&quaternionsUsed[idxLowestQ[iLowest]*4], &coor[i*3], &minEnergyCoor[i*3]);
    }
  
    float minEnergyMinX = minEnergyCoor[0];
    float minEnergyMinY = minEnergyCoor[1];
    float minEnergyMinZ = minEnergyCoor[2];
    for(int i = 1; i < nAtom; i++)
    {
      if (minEnergyCoor[i*3+0] < minEnergyMinX) { minEnergyMinX = minEnergyCoor[i*3+0]; }
      if (minEnergyCoor[i*3+1] < minEnergyMinY) { minEnergyMinY = minEnergyCoor[i*3+1]; }
      if (minEnergyCoor[i*3+2] < minEnergyMinZ) { minEnergyMinZ = minEnergyCoor[i*3+2]; }
    }
  
    double minEnergyCoorDouble[nAtom*3];
    for(int i = 0; i < nAtom; i++)
    {
      minEnergyCoorDouble[i*3 + 0] = (double) minEnergyCoor[i*3 + 0];
      minEnergyCoorDouble[i*3 + 1] = (double) minEnergyCoor[i*3 + 1];
      minEnergyCoorDouble[i*3 + 2] = (double) minEnergyCoor[i*3 + 2];
    }
    
    for(int i = 0; i < nAtom; i++)
    {
      minEnergyCoorDouble[i*3 + 0] += (xmin - minEnergyMinX + minEnergyIdxX[idxLowestQ[iLowest]] * spacing);
      minEnergyCoorDouble[i*3 + 1] += (ymin - minEnergyMinY + minEnergyIdxY[idxLowestQ[iLowest]] * spacing);
      minEnergyCoorDouble[i*3 + 2] += (zmin - minEnergyMinZ + minEnergyIdxZ[idxLowestQ[iLowest]] * spacing);
    }
    
    mol.SetCoordinates(minEnergyCoorDouble);
    fileName = "TranRotaSearch_";
    fileName += std::to_string(iLowest);
    fileName += ".pdb";
    conv.WriteFile(&mol, fileName);
    std::cout << "MinEnergy_" << iLowest << ":" << minEnergyAllQ[idxLowestQ[iLowest]] << std::endl;
  }
  return 0;
}
