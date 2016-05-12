#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>

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
#include "GetIdxOfAtomsForVdwRadius.h"
#include "FillLigandGrid.h"
#include "GeneConformations.h"

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

__global__ void ConjMult(cufftComplex *d_potential_F, cufftComplex *d_ligand_F, int odist, int numOfGridsUsed)
{
  int dist = odist * numOfGridsUsed;
  int idx_l = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int idx_p = idx_l % dist;
  float x = d_ligand_F[idx_l].x;
  float y = d_ligand_F[idx_l].y;
  d_ligand_F[idx_l].x = x * d_potential_F[idx_p].x + y * d_potential_F[idx_p].y;
  d_ligand_F[idx_l].y = x * d_potential_F[idx_p].y - y * d_potential_F[idx_p].x;
};

__global__ void SumGrids(cufftComplex *d_ligand_F, cufftComplex *d_ligand_sum_F, int numOfGridsUsed, int odist, int idist)
{
  int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int idxQuaternion = idx / odist;
  int idxValue = idx % odist;
  d_ligand_sum_F[idx].x = 0;
  d_ligand_sum_F[idx].y = 0;
  for(int i = 0; i < numOfGridsUsed; i++)
  {
    d_ligand_sum_F[idx].x += d_ligand_F[ (idxQuaternion*numOfGridsUsed + i) * odist + idxValue].x;
    d_ligand_sum_F[idx].y += d_ligand_F[ (idxQuaternion*numOfGridsUsed + i) * odist + idxValue].y;
  }
  d_ligand_sum_F[idx].x = d_ligand_sum_F[idx].x / sqrt((float) idist);
  d_ligand_sum_F[idx].y = d_ligand_sum_F[idx].y / sqrt((float) idist);
};

int main(int argc, char** argv)
{
  // read in molecule
  std::string fileName(argv[1]);
  OpenBabel::OBMol mol;
  OpenBabel::OBConversion conv(&std::cin, &std::cout);
  conv.SetInFormat("mol2");
  conv.ReadFile(&mol, fileName);

  // read system
  std::ifstream sysFile;
  sysFile.open(argv[2], std::ifstream::in);
  if (sysFile.fail())
  {
    std::cout << "Open system file failed: " << argv[4] << std::endl;    return 1;
  }
  OpenMM::System *sys = new OpenMM::System();
  sys = OpenMM::XmlSerializer::deserialize<OpenMM::System>(sysFile);

  // random clustered conformations
  double *coorsConformations;
  int numOfConformations;
  numOfConformations = GeneConformations(mol, sys, coorsConformations);
  
  // read coordinate file
  int nAtom = mol.NumAtoms();
  float* coor;
  coor = new float[nAtom*3];
  for(int i = 0; i < nAtom; i++)
  {
    coor[i*3 + 0] = (float) coorsConformations[i*3+0];
    coor[i*3 + 1] = (float) coorsConformations[i*3+1];
    coor[i*3 + 2] = (float) coorsConformations[i*3+2];
    std::cout << coor[i*3 + 0] << "," << coor[i*3 + 1] << "," << coor[i*3 + 2] << std::endl;
  }

  
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
  

  // read quaternioins
  int nQuaternions = atoi(argv[4]);
  float *quaternioins;
  quaternioins = new float[nQuaternions * 4];
  ReadQuaternions(nQuaternions, quaternioins, argv[5]);

  // rotate
  float *coors;
  coors = new float[nQuaternions*nAtom*3];
  for(int i = 0; i < nQuaternions; i++)
  {
    for(int j = 0; j < nAtom; j++)
    {
      Rotate(&quaternioins[i*4], &coor[j*3], &coors[i*nAtom*3+j*3]);
    }
  }

  // get nonbonded parameters
  float atomCharges[nAtom];
  float atomEpsilons[nAtom];
  float atomRadii[nAtom];
  GetNonbondedParameters(sys, atomCharges, atomEpsilons, atomRadii);

  // get index of atoms for each vdw radius
  int numOfVdwGridsUsed;
  std::vector<int> idxOfVdwUsed;
  std::vector< std::vector<int> > idxOfAtomVdwRadius(numOfVdwGrids);
  GetIdxOfAtomsForVdwRadius(nAtom, atomRadii,
  			    numOfVdwGrids, gridRadii,
  			    numOfVdwGridsUsed, idxOfVdwUsed,
  			    idxOfAtomVdwRadius);
  int numOfGridsUsed = numOfVdwGridsUsed + 1;
  
  // calculate minimum coor for each quaternions
  float mincoors[nQuaternions*3];
  GetMinCoors(nQuaternions, nAtom, coors, mincoors);

  // fill ligand grid
  float *ligandGridValues;
  ligandGridValues = new float[nQuaternions*numOfGridsUsed*xdim*ydim*zdim];
  FillLigandGrid(nQuaternions,
  		 nAtom, coors, mincoors,
  		 atomCharges, atomEpsilons,
  		 numOfVdwGridsUsed, idxOfVdwUsed,
  		 idxOfAtomVdwRadius,
  		 xdim, ydim, zdim,
  		 spacing, ligandGridValues);

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
    std::cout << "plan creat failed!";
    return 1;
  }
  potentialRes = cufftExecR2C(potentialPlan, d_potential_f, d_potential_F);
  if (potentialRes != CUFFT_SUCCESS)
  {
    std::cout << "transform failed!";
    return 1;
  }

  // batch cudaFFT for ligand grid
  int nBatchLigand = nQuaternions*numOfGridsUsed;
  cufftReal* d_ligand_f;
  cudaMalloc((void **)&d_ligand_f, sizeof(cufftReal)*nBatchLigand*idist);
  cudaMemcpy(d_ligand_f, ligandGridValues,
  	     sizeof(cufftReal)*nBatchLigand*idist,
  	     cudaMemcpyHostToDevice);
  cufftComplex * d_ligand_F;
  cudaMalloc((void **)&d_ligand_F, sizeof(cufftComplex)*nBatchLigand*odist);
  cufftHandle ligandPlan;
  cufftResult ligandRes = cufftPlanMany(&ligandPlan, 3, n,
  					inembed, istride, idist,
  					onembed, ostride, odist,
  					CUFFT_R2C, nBatchLigand);
  if (ligandRes != CUFFT_SUCCESS)
  {
    std::cout << "plan creat failed!";
    return 1;
  }
  ligandRes = cufftExecR2C(ligandPlan, d_ligand_f, d_ligand_F);
  if (ligandRes != CUFFT_SUCCESS)
  {
    std::cout << "transform failed!";
    return 1;
  }

  // calcualte energy using reverse FFT
  dim3 threads_ConjMult(1024, 1, 1);
  dim3 blocks_ConjMult((nQuaternions*numOfGridsUsed*odist)/(1024*1025) + 1,1024,1);
  ConjMult <<<blocks_ConjMult, threads_ConjMult>>> (d_potential_F, d_ligand_F, odist, numOfGridsUsed);
  CUDA_CHECK();

  cufftComplex * d_ligand_sum_F;
  cudaMalloc((void **)&d_ligand_sum_F, sizeof(cufftComplex)*nQuaternions*odist);
  dim3 threads_SumGrids(1024, 1, 1);
  dim3 blocks_SumGrids((nQuaternions*odist)/(1024*1024),1024,1);
  SumGrids <<<blocks_SumGrids, threads_SumGrids>>> (d_ligand_F, d_ligand_sum_F, numOfGridsUsed, odist, idist);
  CUDA_CHECK();

  cufftReal *d_ligand_sum_f;
  cudaMalloc((void **)&d_ligand_sum_f, sizeof(cufftReal)*nQuaternions*idist);
  cufftHandle ligandRPlan;
  cufftResult ligandRRes = cufftPlanMany(&ligandRPlan, 3, n,
  					 onembed, ostride, odist,
  					 inembed, istride, idist,
  					 CUFFT_C2R, nQuaternions);
  if (ligandRRes != CUFFT_SUCCESS)
  {
    std::cout << "plan creat failed!";
    return 1;
  }
  ligandRRes = cufftExecC2R(ligandRPlan, d_ligand_sum_F, d_ligand_sum_f);
  if (ligandRRes != CUFFT_SUCCESS)
  {
    std::cout << "transform failed!";
    return 1;
  }

  // copy energy back
  float* energy;
  energy = new float[nQuaternions*idist];
  cudaMemcpy(energy, d_ligand_sum_f, sizeof(float)*nQuaternions*idist,
  	     cudaMemcpyDeviceToHost);

  std::cout << "Energy: " << energy[0]/sqrt(idist) << std::endl;
  std::cout << "Energy: " << energy[1]/sqrt(idist) << std::endl;
  
  return 0;
}




