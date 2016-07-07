#include "GeneRandomConformations.h"

#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "OpenMM.h"

#include <openbabel/obconversion.h>
#include <openbabel/rotor.h>
#include <openbabel/rotamer.h>
#include <openbabel/mol.h>

int GeneRandomConformations(OpenBabel::OBMol &mol, OpenMM::System *sys, int maxNumOfConformations, double* &coorsConformers)
{
  // check if the mol has rotatable bonds
  OpenBabel::OBRotorList rotors; // rotatable bonds 
  bool ifHasRotors;
  ifHasRotors = rotors.Setup(mol);
  if (! ifHasRotors) // this is a rigid ligand
  {
    std::cout << "no rotatable bonds found in molecule" << std::endl;
    coorsConformers = new double[mol.NumAtoms()*3];
    memcpy(coorsConformers, mol.GetCoordinates(), sizeof(double)*mol.NumAtoms()*3);
    return 1;
  }

  //// When the ligand has rotatable bonds
  ////  generate random conformers  
  int N = 2000;  // num of random conformers from which not too high energy poses are selected
  OpenBabel::OBMol mols[N];
  for(int i = 0; i < N; i++)
  {
    mols[i] = mol;
  }
  
  OpenBabel::OBRotorIterator rotorIter;
  for(int i = 0; i < N; i++)
  {
    rotors.Setup(mols[i]);
    for(rotorIter = rotors.BeginRotors(); rotorIter != rotors.EndRotors(); rotorIter++)
    {
      double tmp = M_PI * (rand() / ((float) RAND_MAX) - 0.5) * 2;
      (*rotorIter)->SetToAngle(mols[i].GetCoordinates(), tmp);
    }
  }
  std::cout << "random  rotor done" << std::endl;
  
  // brief minimization and calculate energy for each conformation
  OpenMM::VerletIntegrator integrator(0.001);
  OpenMM::LocalEnergyMinimizer minimizer;
  // OpenMM::Platform::loadPluginLibrary("/home/xqding/apps/openmmDev/lib/plugins/libOpenMMCPU.so");
  OpenMM::Platform& platform = OpenMM::Platform::getPlatformByName("Reference");
  OpenMM::Context context(*sys, integrator, platform);
  printf( "REMARK  GeneConformation Using OpenMM platform %s\n",
	  context.getPlatform().getName().c_str() );
  OpenMM::State state;
  std::vector<OpenMM::Vec3> position(sys->getNumParticles());
  std::vector<double> energy(N,0);
  for (int k = 0; k < N; k++)
  {
    for(int i = 0; i < sys->getNumParticles(); i++)
    {
      position[i] = OpenMM::Vec3(mols[k].GetCoordinates()[i*3+0]*OpenMM::NmPerAngstrom,
				 mols[k].GetCoordinates()[i*3+1]*OpenMM::NmPerAngstrom,
				 mols[k].GetCoordinates()[i*3+2]*OpenMM::NmPerAngstrom);
    }
    context.setPositions(position);
    minimizer.minimize(context, 0.01, 100);
    state = context.getState(OpenMM::State::Positions | OpenMM::State::Energy);
    energy[k] = state.getPotentialEnergy()*OpenMM::KcalPerKJ;
    position = state.getPositions();
    for(int i = 0; i < sys->getNumParticles(); i++)
    {
      mols[k].GetCoordinates()[i*3+0] = position[i][0] * OpenMM::AngstromsPerNm;
      mols[k].GetCoordinates()[i*3+1] = position[i][1] * OpenMM::AngstromsPerNm;
      mols[k].GetCoordinates()[i*3+2] = position[i][2] * OpenMM::AngstromsPerNm;
    }    
  }
  // copy out the random conformers with energy less than 100 Kcal/Mol
  int numOfSmallEnergyPose = 0;
  for(int i = 0; i < N; i++)
  {
    if (energy[i] < 1000)
    {
      numOfSmallEnergyPose++;
    }
  }

  if (numOfSmallEnergyPose > maxNumOfConformations)
  {
    numOfSmallEnergyPose = maxNumOfConformations;
  }
  
  coorsConformers = new double[numOfSmallEnergyPose*sys->getNumParticles()*3];
  int j = 0;
  
  for(int i = 0; i < N; i++)
  {
    if (j >= numOfSmallEnergyPose) { break; }
    if (energy[i] < 1000)
    {
      memcpy(&coorsConformers[j*sys->getNumParticles()*3],
	     mols[i].GetCoordinates(),
	     sizeof(double)*sys->getNumParticles()*3);
      j++;
    }

  }
  return numOfSmallEnergyPose;
}
