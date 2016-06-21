#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>
#include <stddef.h>

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
#include "AddGridForcesToOpenMMSystem.h"
#include "FilterQuaternions.h"

#define CUDA_CALL(F)  if( (F) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__); exit(-1);} 

#define CUDA_CHECK()  if( (cudaPeekAtLastError()) != cudaSuccess ) \
  {printf("Error %s at %s:%d\n", cudaGetErrorString(cudaGetLastError()), \
   __FILE__,__LINE__-1); exit(-1);} 

//// main function ////
// Usage:
// TranRotaConfSearch .mol2 sys.xml grid.txt maxNumOfConf numOfRotaPerConf maxNumOfRotaPerConf numOfRotaSample nLowest mode
// maxNumOfConf: maximum num of random conformations
// numOfRotatPerConf: num of directions for each coforamtion
// mode:
// - 0: only search translation
// - 1: only search translation and rotation. The conforamtion is given in mol2 file
// - 2: search translation, rotation, and conformation

int main(int argc, char** argv)
{
  OpenMM::Platform::loadPluginsFromDirectory(
  					     "/home/xqding/apps/openmmDev/lib/plugins");

  // parse the command line parameters
  std::string mol2FileName(argv[1]);
  std::string sysFileName(argv[2]);
  std::string gridFileName(argv[3]);
  int maxNumOfConformations = atoi(argv[4]);
  int numOfRotaPerConformation = atoi(argv[5]);
  int maxNumOfRotaPerConf = atoi(argv[6]);
  int numOfRotaSample = atoi(argv[7]);
  int nLowest = atoi(argv[8]);
  int mode = atoi(argv[9]);
  if (mode == 0)
  {
    std::cout << "Model 0: only search for translation" << std::endl;
  }
  else if (mode == 1)
  {
    std::cout << "Model 1: search for translation and rotation" << std::endl;
  }
  else if (mode == 2)
  {
    std::cout << "Model 2: search for translation, rotation and conformation" << std::endl;
  }
  else
  {
    std::cout << "Error: The value of model has to be 0, 1 or 2" << std::endl;
    return 1;
  }

  // read molecule
  OpenBabel::OBMol mol;
  OpenBabel::OBConversion conv(&std::cin, &std::cout);
  conv.SetInFormat("mol2");
  conv.SetOutFormat("pdb");
  conv.ReadFile(&mol, mol2FileName);
  int nAtom = mol.NumAtoms();
  
  // read system
  std::ifstream sysFile;
  sysFile.open(sysFileName, std::ifstream::in);
  if (sysFile.fail())
  {
    std::cout << "Open system file failed: " << sysFileName << std::endl;
    return 1;
  }
  OpenMM::System *sys = new OpenMM::System();
  sys = OpenMM::XmlSerializer::deserialize<OpenMM::System>(sysFile);

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
  	    gridFileName);
  int numOfVdwGrids = numOfGrids - 1;

  // random clustered conformations
  double *coorsConformations;
  int numOfConformations;
  if (mode == 0 || mode == 1) // use one conformatoin in mol file
  {
    coorsConformations = new double [nAtom * 3];
    memcpy(coorsConformations, mol.GetCoordinates(), sizeof(double) * nAtom * 3);
    numOfConformations = 1;
  }
  if (mode == 2)
  {
    numOfConformations = GeneConformations(mol, sys, maxNumOfConformations, coorsConformations);
  }
  std::cout << "num of conformations: " << numOfConformations << std::endl;
  
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

  // add grid forces to OpenMM System via cumstomized forces
  double gridMinX = midx - xlen / 2;
  double gridMinY = midy - ylen / 2;
  double gridMinZ = midz - zlen / 2;
  double gridMaxX = gridMinX + (xdim - 1) * spacing;
  double gridMaxY = gridMinY + (ydim - 1) * spacing;
  double gridMaxZ = gridMinZ + (zdim - 1) * spacing;
  AddGridForcesToOpenMMSystem(xdim, ydim, zdim,
  			      gridMinX, gridMinY, gridMinZ,
  			      gridMaxX, gridMaxY, gridMaxZ,
  			      numOfVdwGridsUsed, usedGridValues,
  			      idxOfVdwUsed,
  			      idxOfAtomVdwRadius,
  			      &usedGridValues[numOfVdwGridsUsed*xdim*ydim*zdim],
  			      sys
  			      );

  // build OpenMM context
  OpenMM::LangevinIntegrator integrator(300, 10, 0.0015);
  // OpenMM::VerletIntegrator integrator(0.001);
  OpenMM::LocalEnergyMinimizer minimizer;
  
  OpenMM::Context context(*sys, integrator);
  printf( "REMARK  Add custom force Using OpenMM platform %s\n",
  	  context.getPlatform().getName().c_str() );
  OpenMM::State state;
  std::vector<OpenMM::Vec3> position(sys->getNumParticles());
  
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
  
  // generate uniform quaternions and setup data structure for all quaternions
  float* quaternions = 0;
  std::random_device rd;
  std::mt19937_64 gen(rd());
  
  // for one batch of quaternions
  int numOfQuaternionsOneBatch = 80;
  int numOfBatches = 0;

  // ligand grid for one batch
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
    std::cout << "plan creat failed!";
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
    std::cout << "plan creat failed!";
    return 1;
  }

  // host energy 
  float* energy;
  energy = new float[numOfQuaternionsOneBatch*idist];

  // coordinates for one conformation
  float* coor;
  coor = new float[nAtom*3];
  
  // ignore quaterions, whose end structures' dimenstion is larger than the grids
  size_t maxNQuaternionsUsed = maxNumOfConformations * numOfRotaPerConformation / numOfConformations + 1;
  if (maxNQuaternionsUsed > maxNumOfRotaPerConf)
  {
    maxNQuaternionsUsed = maxNumOfRotaPerConf;
  }
  
  size_t numOfQuaternionsUsed;
  float* quaternionsUsed = 0;  

  int *minEnergyIdxX = 0;
  int *minEnergyIdxY = 0;
  int *minEnergyIdxZ = 0;

  float *coorsUsed = 0;
  float *mincoorsUsed = 0;
  float *maxcoorsUsed = 0;
  float *ligandLengthUsed = 0;
  
  // mol for saving lowest energy pose
  OpenBabel::OBMol finalPoses[numOfConformations * nLowest];
  double energyOfFinalPoses[numOfConformations * nLowest];
  
  for(int i = 0; i < numOfConformations * nLowest; i++)
  {
    finalPoses[i] = mol;
  }
  
  // file for saving energy values of end poses 
  std::ofstream energyFile("energy.txt", std::ofstream::out);

  for (int idxOfConformer = 0; idxOfConformer < numOfConformations; idxOfConformer++)
  {
    std::cout << "idxOfConformer: " << idxOfConformer << std::endl;
    // get coordinates for one conformer
    for(int i = 0; i < nAtom; i++)
    {
      coor[i*3 + 0] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 0];
      coor[i*3 + 1] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 1];
      coor[i*3 + 2] = (float) coorsConformations[(idxOfConformer*nAtom + i)*3 + 2];
    }

    // generate all quaternions and ignore some quaternions, which will rotate the ligand
    // to have larger dimension than the grid
    if (mode == 0)
    {
      quaternions = new float[numOfRotaSample * 4]; 
      for(int i = 0; i < numOfRotaSample; i++)
      {
	quaternions[i*4 + 0] = 1;
	quaternions[i*4 + 1] = 0;
	quaternions[i*4 + 2] = 0;
	quaternions[i*4 + 3] = 0;
      }
    }
    if (mode == 1 || mode == 2)
    {
      QuaternionUniformSampling(gen, quaternions, numOfRotaSample);
    }
    
    numOfQuaternionsUsed = FilterQuaternions(coor, nAtom,
  					     numOfRotaSample, quaternions,
  					     xlen, ylen, zlen,
  					     maxNQuaternionsUsed, quaternionsUsed);

    delete[] minEnergyIdxX;
    delete[] minEnergyIdxY;
    delete[] minEnergyIdxZ;
    minEnergyIdxX = new int[numOfQuaternionsUsed];
    minEnergyIdxY = new int[numOfQuaternionsUsed];
    minEnergyIdxZ = new int[numOfQuaternionsUsed];
    std::vector <float> minEnergyQuaternionsUsed(numOfQuaternionsUsed);
    for(int i = 0; i < numOfQuaternionsUsed; i++)
    {
      minEnergyQuaternionsUsed[i] = INFINITY;
    }

    delete[] coorsUsed;
    delete[] mincoorsUsed;
    delete[] maxcoorsUsed;
    delete[] ligandLengthUsed;
    coorsUsed = new float[numOfQuaternionsUsed*nAtom*3];
    mincoorsUsed = new float[numOfQuaternionsUsed*3];
    maxcoorsUsed = new float[numOfQuaternionsUsed*3];
    ligandLengthUsed = new float[numOfQuaternionsUsed*3];

    for(int i = 0; i < numOfQuaternionsUsed; i++)
    {
      for(int j = 0; j < nAtom; j++)
      {
    	Rotate(&quaternionsUsed[i*4], &coor[j*3], &coorsUsed[i*nAtom*3+j*3]);
      }
    }
    
    // calculate minimum coor for each quaternions
    GetMinCoors(numOfQuaternionsUsed, nAtom, coorsUsed, mincoorsUsed);
    
    // calculate maximum coor for each quaternions
    GetMaxCoors(numOfQuaternionsUsed, nAtom, coorsUsed, maxcoorsUsed);

    // calculate the length for each quaternion
    for(int i = 0; i < numOfQuaternionsUsed; i++)
    {
      ligandLengthUsed[i*3 + 0] = maxcoorsUsed[i*3 + 0] - mincoorsUsed[i*3 + 0];
      ligandLengthUsed[i*3 + 1] = maxcoorsUsed[i*3 + 1] - mincoorsUsed[i*3 + 1];
      ligandLengthUsed[i*3 + 2] = maxcoorsUsed[i*3 + 2] - mincoorsUsed[i*3 + 2];
    }

    // loop over batches of quaternions
    // num of batches
    if (numOfQuaternionsUsed % numOfQuaternionsOneBatch == 0)
    {
      numOfBatches = numOfQuaternionsUsed / numOfQuaternionsOneBatch;
    }
    else
    {
      numOfBatches = numOfQuaternionsUsed / numOfQuaternionsOneBatch + 1;
    }
    
    for(int idxOfBatch = 0; idxOfBatch < numOfBatches; idxOfBatch++)
    {
      std::cout << "idxOfBatch: " << idxOfBatch << std::endl;  
      // fill ligand grid
      memset(ligandGridValues, 0, sizeof(float)*numOfQuaternionsOneBatch*numOfGridsUsed*xdim*ydim*zdim);
      if ((idxOfBatch + 1) * numOfQuaternionsOneBatch > numOfQuaternionsUsed)
      {
  	FillLigandGrid(numOfQuaternionsUsed - idxOfBatch * numOfQuaternionsOneBatch,
  		       nAtom, &coorsUsed[idxOfBatch*numOfQuaternionsOneBatch*nAtom*3], &mincoorsUsed[idxOfBatch*numOfQuaternionsOneBatch*3],
  		       atomCharges, atomEpsilons,
  		       numOfVdwGridsUsed, idxOfVdwUsed,
  		       idxOfAtomVdwRadius,
  		       xdim, ydim, zdim,
  		       spacing, ligandGridValues);
      }
      else
      {	            
  	FillLigandGrid(numOfQuaternionsOneBatch,
  		       nAtom, &coorsUsed[idxOfBatch*numOfQuaternionsOneBatch*nAtom*3], &mincoorsUsed[idxOfBatch*numOfQuaternionsOneBatch*3],
  		       atomCharges, atomEpsilons,
  		       numOfVdwGridsUsed, idxOfVdwUsed,
  		       idxOfAtomVdwRadius,
  		       xdim, ydim, zdim,
  		       spacing, ligandGridValues);
      }

      // batch cudaFFT for ligand grid
      cudaMemcpy(d_ligand_f, ligandGridValues,
    		 sizeof(cufftReal)*nBatchLigand*idist,
    		 cudaMemcpyHostToDevice);
      ligandRes = cufftExecR2C(ligandPlan, d_ligand_f, d_ligand_F);
      if (ligandRes != CUFFT_SUCCESS)
      {
    	std::cout << "ligand grid transform failed!";
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
    	std::cout << "ligand grid reverse transform failed!";
    	return 1;
      }
      
      // copy energy back
      cudaMemcpy(energy, d_ligand_sum_f, sizeof(float)*numOfQuaternionsOneBatch*idist,
    		 cudaMemcpyDeviceToHost);

      // record the minimum energy pose in terms of quaternions, x, y and z
      for(int q = 0; q < numOfQuaternionsOneBatch; q++)
      {
    	int idxOfQuaternions = idxOfBatch * numOfQuaternionsOneBatch + q;
  	if(idxOfQuaternions < numOfQuaternionsUsed)
  	{
  	  for(int i = 0; i < (xdim-int(ligandLengthUsed[idxOfQuaternions*3+0]/spacing)-2); i++)
  	  {
  	    for(int j = 0; j < (ydim-int(ligandLengthUsed[idxOfQuaternions*3+1]/spacing)-2); j++)
  	    {
  	      for(int k = 0; k < (ydim-int(ligandLengthUsed[idxOfQuaternions*3+2]/spacing)-2); k++)
  	      {
  		int tmp = q*idist + (i*ydim + j)*zdim + k;
    		if(energy[tmp] / sqrt(idist) < minEnergyQuaternionsUsed[idxOfQuaternions])
    		{
    		  minEnergyQuaternionsUsed[idxOfQuaternions] = energy[tmp] / sqrt(idist);
    		  minEnergyIdxX[idxOfQuaternions] = i;
    		  minEnergyIdxY[idxOfQuaternions] = j;
    		  minEnergyIdxZ[idxOfQuaternions] = k;
    		}
    	      }
    	    }
    	  }
    	}
      }      
    } // finish all batches of quaternions for one conformer
    
    // calculate the coordinates corresponding to lowest nLowest energy orientation
    std::vector<size_t> idxOfSortedQuater;
    idxOfSortedQuater = sort_index<float>(minEnergyQuaternionsUsed);
    for(int iLowest = 0; iLowest < nLowest && iLowest < numOfQuaternionsUsed; iLowest++)
    {
      int idxQ = idxOfSortedQuater[iLowest];
      double minEnergyCoorDouble[nAtom*3];
      for(int i = 0; i < nAtom; i++)
      {
    	minEnergyCoorDouble[i*3 + 0] = (double) coorsUsed[idxQ*nAtom*3 + i*3 + 0];
    	minEnergyCoorDouble[i*3 + 1] = (double) coorsUsed[idxQ*nAtom*3 + i*3 + 1];
    	minEnergyCoorDouble[i*3 + 2] = (double) coorsUsed[idxQ*nAtom*3 + i*3 + 2];
      }
    
      for(int i = 0; i < nAtom; i++)
      {
    	minEnergyCoorDouble[i*3 + 0] += (gridMinX - mincoorsUsed[idxQ*3 + 0] + minEnergyIdxX[idxQ] * spacing);
    	minEnergyCoorDouble[i*3 + 1] += (gridMinY - mincoorsUsed[idxQ*3 + 1] + minEnergyIdxY[idxQ] * spacing);
    	minEnergyCoorDouble[i*3 + 2] += (gridMinZ - mincoorsUsed[idxQ*3 + 2] + minEnergyIdxZ[idxQ] * spacing);
      }

      for(int i = 0; i < sys->getNumParticles(); i++)
      {
      	position[i] = OpenMM::Vec3(minEnergyCoorDouble[i*3+0]*OpenMM::NmPerAngstrom,
      				   minEnergyCoorDouble[i*3+1]*OpenMM::NmPerAngstrom,
      				   minEnergyCoorDouble[i*3+2]*OpenMM::NmPerAngstrom);
      }
      
      context.setPositions(position);
      if (mode == 2)
      {
	// simulated annealing
	for(int i = 700; i >= 100; i-=30)
	{
	  integrator.setTemperature(i);
	  integrator.step(500);
	}
	// minimize at the end
	minimizer.minimize(context, 0.001, 100);
      }
      // state = context.getState(OpenMM::State::Energy, false, 1<<10 | 1<<11);
      state = context.getState(OpenMM::State::Energy);
      for(int i = 0; i < sys->getNumParticles(); i++)
      {
  	minEnergyCoorDouble[i*3 + 0] = position[i][0] * OpenMM::AngstromsPerNm;
  	minEnergyCoorDouble[i*3 + 1] = position[i][1] * OpenMM::AngstromsPerNm;
  	minEnergyCoorDouble[i*3 + 2] = position[i][2] * OpenMM::AngstromsPerNm;
      }

      finalPoses[idxOfConformer * nLowest + iLowest].SetCoordinates(minEnergyCoorDouble);
      energyOfFinalPoses[idxOfConformer * nLowest + iLowest] = state.getPotentialEnergy() * OpenMM::KcalPerKJ;
      
      // mol.SetCoordinates(minEnergyCoorDouble);
      std::string fileName;
      fileName = "conformer_";
      fileName += std::to_string(idxOfConformer);
      fileName += "_";
      fileName += std::to_string(iLowest);
      fileName += ".pdb";
      // conv.WriteFile(&mol, fileName);
      conv.WriteFile(&finalPoses[idxOfConformer*nLowest+iLowest], fileName);
      energyFile << fileName << ","
      		 << idxOfConformer << ","
      		 << iLowest << ","
      		 << state.getPotentialEnergy() * OpenMM::KcalPerKJ
      		 << std::endl;
      std::cout << "Conformer: " << idxOfConformer
      		<< ", IdxQ: " << idxQ
      		<< ", IdxX: " << minEnergyIdxX[idxQ]
      		<< ", IdxY: " << minEnergyIdxY[idxQ]
      		<< ", IdxZ: " << minEnergyIdxZ[idxQ]
      		<< ", MinEnergyTranRota:" << minEnergyQuaternionsUsed[idxQ]
      		<< ", Potential Energy: " << state.getPotentialEnergy() * OpenMM::KcalPerKJ << std::endl;
    }
  }
  energyFile.close();
  return 0;
}
