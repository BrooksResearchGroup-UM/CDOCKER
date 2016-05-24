#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>

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
  int nAtom = mol.NumAtoms();
  
  // read system
  std::ifstream sysFile;
  sysFile.open(argv[2], std::ifstream::in);
  if (sysFile.fail())
  {
    std::cout << "Open system file failed: " << argv[4] << std::endl;    return 1;
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
  std::vector <double> tmpGridValue(xdim*ydim*zdim, 0);
  for(int i = 0; i < xdim*ydim*zdim; i++)
  {
    tmpGridValue[i] = gridValues[numOfVdwGrids*xdim*ydim*zdim + i];
  }
  OpenMM::Continuous3DFunction *elecGridFunction =
    new OpenMM::Continuous3DFunction(xdim, ydim, zdim,
				     tmpGridValue,
				     xmin, xmax,
				     ymin, ymax,
				     zmin, zmax);
  
  OpenMM::CustomCompoundBondForce *elecGridPotential =
    new OpenMM::CustomCompoundBondForce(1, "elecGrid(x1,y1,z1) * q");
  sys->addForce(elecGridPotential);
  elecGridPotential->addTabulatedFunction("elecGrid", elecGridFunction);
  elecGridPotential->addPerBondParameter("q");

  std::vector<int> idxParticle(1,0);
  std::vector<double> parameter(1,0);
  for (int i = 0; i < sys->getNumParticles(); i++)
  {
    idxParticle[0] = i;
    parameter[0] = atomCharges[i];
    elecGridPotential->addBond(idxParticle, parameter);
  }

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

  // add vdw grid force
  OpenMM::Continuous3DFunction *vdwGridFunctions[numOfVdwGridsUsed];
  OpenMM::CustomCompoundBondForce *vdwGridPotentials[numOfVdwGridsUsed];

  std::string formula;
  for(int k = 0; k < numOfVdwGridsUsed; k++)
  {
    for(int i = 0; i < xdim*ydim*zdim; i++)
    {
      tmpGridValue[i] = gridValues[k*xdim*ydim*zdim + i];
    }    
    vdwGridFunctions[k] =  new OpenMM::Continuous3DFunction(xdim, ydim, zdim,
							    tmpGridValue,
							    xmin, xmax,
							    ymin, ymax,
							    zmin, zmax);
    formula = "vdwGrid";
    formula += std::to_string(k);
    formula += "(x1,y1,z1) * sqrt(epsilon)";
    vdwGridPotentials[k] = new OpenMM::CustomCompoundBondForce(1, formula);
    sys->addForce(vdwGridPotentials[k]);

    formula = "vdwGrid";
    formula += std::to_string(k);
    elecGridPotential->addTabulatedFunction(formula, vdwGridFunctions[k]);
    elecGridPotential->addPerBondParameter("epsilon");

    for (int i = 0; i < idxOfAtomVdwRadius[idxOfVdwUsed[k]].size(); i++)
    {
      int idx = idxOfAtomVdwRadius[idxOfVdwUsed[k]][i];
      idxParticle[0] = idx;
      parameter[0] = atomEpsilons[idx];
      elecGridPotential->addBond(idxParticle, parameter);
    }    
  }

  
  /* // batch cudaFFT for potential grids */
  /* int n[3]; */
  /* n[0] = xdim; */
  /* n[1] = ydim; */
  /* n[2] = zdim; */
  /* int inembed[3]; */
  /* inembed[0] = xdim; */
  /* inembed[1] = ydim; */
  /* inembed[2] = zdim; */
  /* int idist = inembed[0] * inembed[1] * inembed[2]; */
  /* int istride = 1; */
  
  /* int onembed[3]; */
  /* onembed[0] = xdim; */
  /* onembed[1] = ydim; */
  /* onembed[2] = zdim/2 + 1; */
  /* int odist = onembed[0] * onembed[1] * onembed[2]; */
  /* int ostride = 1; */
  /* int nBatchPotential = numOfGridsUsed; */
  
  /* cufftReal* d_potential_f; */
  /* cudaMalloc((void **)&d_potential_f, sizeof(cufftReal)*nBatchPotential*idist); */
  /* cudaMemcpy(d_potential_f, usedGridValues, */
  /* 	     sizeof(cufftReal)*nBatchPotential*idist, */
  /* 	     cudaMemcpyHostToDevice); */
  /* cufftComplex *d_potential_F; */
  /* cudaMalloc((void **)&d_potential_F, sizeof(cufftComplex)*nBatchPotential*odist); */
  /* cufftHandle potentialPlan; */
  /* cufftResult potentialRes = cufftPlanMany(&potentialPlan, 3, n, */
  /* 					   inembed, istride, idist, */
  /* 					   onembed, ostride, odist, */
  /* 					   CUFFT_R2C, nBatchPotential); */
  /* if (potentialRes != CUFFT_SUCCESS) */
  /* { */
  /*   std::cout << "plan creat failed!"; */
  /*   return 1; */
  /* } */
  /* potentialRes = cufftExecR2C(potentialPlan, d_potential_f, d_potential_F); */
  /* if (potentialRes != CUFFT_SUCCESS) */
  /* { */
  /*   std::cout << "transform failed!"; */
  /*   return 1; */
  /* } */

  /* // read quaternioins */
  /* int numOfTotalQuaternions = atoi(argv[4]); */
  /* float *quaternioins; */
  /* quaternioins = new float[numOfTotalQuaternions * 4]; */
  /* ReadQuaternions(numOfTotalQuaternions, quaternioins, argv[5]); */

  /* // random clustered conformations */
  /* double *coorsConformations; */
  /* int numOfConformations; */
  /* numOfConformations = GeneConformations(mol, sys, coorsConformations); */

  /* // loop over all batches for different orientation */
  /* int numOfQuaternionsOneBatch = 30; */
  /* int numOfBatches = numOfTotalQuaternions / numOfQuaternionsOneBatch + 1; */
    
  /* // allocate the data structures which will be used */
  /* float *coors; // rotated coordinates */
  /* coors = new float[numOfQuaternionsOneBatch*nAtom*3]; */
  /* float mincoors[numOfQuaternionsOneBatch*3]; // minimium coordinates along x, y, and z */
  /* float maxcoors[numOfQuaternionsOneBatch*3]; // maximum coordinates along x, y, and z */
  /* float ligandLength[numOfQuaternionsOneBatch*3]; // lenth along x, y and z for each orientation */
  
  /* float *ligandGridValues; // grid for ligand */
  /* ligandGridValues = new float[numOfQuaternionsOneBatch*numOfGridsUsed*xdim*ydim*zdim]; */
  /* // cudaFFT for ligand grid */
  /* int nBatchLigand = numOfQuaternionsOneBatch*numOfGridsUsed; */
  /* cufftReal* d_ligand_f; */
  /* cudaMalloc((void **)&d_ligand_f, sizeof(cufftReal)*nBatchLigand*idist); */
  /* cufftComplex * d_ligand_F; */
  /* cudaMalloc((void **)&d_ligand_F, sizeof(cufftComplex)*nBatchLigand*odist); */
  /* cufftHandle ligandPlan; */
  /* cufftResult ligandRes = cufftPlanMany(&ligandPlan, 3, n, */
  /* 					inembed, istride, idist, */
  /* 					onembed, ostride, odist, */
  /* 					CUFFT_R2C, nBatchLigand); */
  /* if (ligandRes != CUFFT_SUCCESS) */
  /* { */
  /*   std::cout << "plan creat failed!"; */
  /*   return 1; */
  /* } */

  /* dim3 threads_ConjMult(1024, 1, 1); */
  /* dim3 blocks_ConjMult((numOfQuaternionsOneBatch*numOfGridsUsed*odist)/(1024*1025) + 1,1024,1); */
  /* cufftComplex * d_ligand_sum_F; */
  /* cudaMalloc((void **)&d_ligand_sum_F, sizeof(cufftComplex)*numOfQuaternionsOneBatch*odist); */
  /* dim3 threads_SumGrids(1024, 1, 1); */
  /* dim3 blocks_SumGrids((numOfQuaternionsOneBatch*odist)/(1024*1024),1024,1); */
  /* cufftReal *d_ligand_sum_f; */
  /* cudaMalloc((void **)&d_ligand_sum_f, sizeof(cufftReal)*numOfQuaternionsOneBatch*idist); */
  /* cufftHandle ligandRPlan; */
  /* cufftResult ligandRRes = cufftPlanMany(&ligandRPlan, 3, n, */
  /* 					 onembed, ostride, odist, */
  /* 					 inembed, istride, idist, */
  /* 					 CUFFT_C2R, numOfQuaternionsOneBatch); */
  /* if (ligandRRes != CUFFT_SUCCESS) */
  /* { */
  /*   std::cout << "plan creat failed!"; */
  /*   return 1; */
  /* } */
  /* float* energy; */
  /* energy = new float[numOfQuaternionsOneBatch*idist]; */

  /* // read coordinate file */
  /* float* coor; */
  /* coor = new float[nAtom*3]; */

  /* int minEnergyQ = 0; */
  /* int minEnergyX = 0; */
  /* int minEnergyY = 0; */
  /* int minEnergyZ = 0; */
  /* float minEnergy = INFINITY; */

  /* for (int idxOfConformer = 0; idxOfConformer < numOfConformations; idxOfConformer++) */
  /* { */
  /*   minEnergyQ = 0; */
  /*   minEnergyX = 0; */
  /*   minEnergyY = 0; */
  /*   minEnergyZ = 0; */
  /*   minEnergy = INFINITY; */
    
  /*   std::cout << "idxOfConformer: " << idxOfConformer << std::endl; */
  /*   for(int i = 0; i < nAtom; i++) */
  /*   { */
  /*     coor[i*3 + 0] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 0]; */
  /*     coor[i*3 + 1] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 1]; */
  /*     coor[i*3 + 2] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 2]; */
  /*   } */
  /*   for(int idxOfBatch = 0; idxOfBatch < numOfBatches; idxOfBatch++) */
  /*   { */
  /*     std::cout << "idxOfBatch: " << idxOfBatch << std::endl;  */
  /*     // rotate   */
  /*     for(int i = 0; i < numOfQuaternionsOneBatch; i++) */
  /*     { */
  /* 	if (idxOfBatch*numOfQuaternionsOneBatch + i < numOfTotalQuaternions) */
  /* 	{ */
  /* 	  for(int j = 0; j < nAtom; j++) */
  /* 	  { */
  /* 	    Rotate(&quaternioins[(idxOfBatch*numOfQuaternionsOneBatch + i)*4], &coor[j*3], &coors[i*nAtom*3+j*3]); */
  /* 	  } */
  /* 	} */
  /*     } */
    
  /*     // calculate minimum coor for each quaternions */
  /*     GetMinCoors(numOfQuaternionsOneBatch, nAtom, coors, mincoors); */

  /*     // calculate maximum coor for each quaternions */
  /*     GetMaxCoors(numOfQuaternionsOneBatch, nAtom, coors, maxcoors); */

  /*     // calculate the length for each quaternion */
  /*     for(int i = 0; i < numOfQuaternionsOneBatch; i++) */
  /*     { */
  /* 	ligandLength[i*3 + 0] = maxcoors[i*3 + 0] - mincoors[i*3 + 0]; */
  /* 	ligandLength[i*3 + 1] = maxcoors[i*3 + 1] - mincoors[i*3 + 1]; */
  /* 	ligandLength[i*3 + 2] = maxcoors[i*3 + 2] - mincoors[i*3 + 2]; */
  /*     } */
      
  /*     // fill ligand grid */
  /*     memset(ligandGridValues, 0, sizeof(float)*numOfQuaternionsOneBatch*numOfGridsUsed*xdim*ydim*zdim); */
  /*     FillLigandGrid(numOfQuaternionsOneBatch, */
  /* 		     nAtom, coors, mincoors, */
  /* 		     atomCharges, atomEpsilons, */
  /* 		     numOfVdwGridsUsed, idxOfVdwUsed, */
  /* 		     idxOfAtomVdwRadius, */
  /* 		     xdim, ydim, zdim, */
  /* 		     spacing, ligandGridValues); */


  /*     // batch cudaFFT for ligand grid */
  /*     cudaMemcpy(d_ligand_f, ligandGridValues, */
  /* 		 sizeof(cufftReal)*nBatchLigand*idist, */
  /* 		 cudaMemcpyHostToDevice); */
  /*     ligandRes = cufftExecR2C(ligandPlan, d_ligand_f, d_ligand_F); */
  /*     if (ligandRes != CUFFT_SUCCESS) */
  /*     { */
  /* 	std::cout << "transform failed!"; */
  /* 	return 1; */
  /*     } */

  /*     // calcualte energy using reverse FFT */
  /*     ConjMult <<<blocks_ConjMult, threads_ConjMult>>> (d_potential_F, d_ligand_F, odist, numOfGridsUsed); */
  /*     CUDA_CHECK(); */

  /*     SumGrids <<<blocks_SumGrids, threads_SumGrids>>> (d_ligand_F, d_ligand_sum_F, numOfGridsUsed, odist, idist); */
  /*     CUDA_CHECK(); */

  /*     ligandRRes = cufftExecC2R(ligandRPlan, d_ligand_sum_F, d_ligand_sum_f); */
  /*     if (ligandRRes != CUFFT_SUCCESS) */
  /*     { */
  /* 	std::cout << "transform failed!"; */
  /* 	return 1; */
  /*     } */

  /*     // copy energy back */
  /*     cudaMemcpy(energy, d_ligand_sum_f, sizeof(float)*numOfQuaternionsOneBatch*idist, */
  /* 		 cudaMemcpyDeviceToHost); */

  /*     // get the index of minimum energy pose */
  /*     for(int q = 0; q < numOfQuaternionsOneBatch; q++) */
  /*     { */
  /*     	for(int i = 0; i < (xdim - int(ligandLength[q*3+0] / spacing) - 2); i++) */
  /*     	{ */
  /*     	  for(int j = 0; j < (ydim - int(ligandLength[q*3+1] / spacing) - 2); j++ ) */
  /*     	  { */
  /*     	    for(int k = 0; k < (zdim - int(ligandLength[q*3+2] / spacing) - 2); k++) */
  /*     	    { */
  /*     	      int tmp = q*idist + (i*ydim + j)*zdim+k; */
  /*     	      if(idxOfBatch*numOfQuaternionsOneBatch*idist + tmp < numOfTotalQuaternions*idist) */
  /*     	      { */
  /*     		if (energy[tmp] / sqrt(idist) < minEnergy) */
  /*     		{ */
  /*     		  minEnergy = energy[tmp] / sqrt(idist); */
  /*     		  minEnergyQ = q; */
  /*     		  minEnergyX = i; */
  /*     		  minEnergyY = j; */
  /*     		  minEnergyZ = k; */
  /*     		} */
  /*     	      } */
  /*     	    } */
  /*     	  } */
  /*     	} */
  /*     } */
  /*   } */
  /*   std::cout << "IdxConformer: " << idxOfConformer << "," ; */
  /*   std::cout << "minEnergyQuaternionIdx: " << minEnergyQ << "," ; */
  /*   std::cout << "minEnergyX: " << minEnergyX << "," ; */
  /*   std::cout << "minEnergyY: " << minEnergyY << "," ; */
  /*   std::cout << "minEnergyZ: " << minEnergyZ << "," ; */
  /*   std::cout << "minEnergy: " << minEnergy << std::endl; */
  /* } */
  return 0;
}
