#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include "OpenMM.h"

#include <openbabel/obconversion.h>
#include <openbabel/rotor.h>
#include <openbabel/rotamer.h>
#include <openbabel/mol.h>
#include <openbabel/math/align.h>
#include "GeneConformations.h"

int GeneConformations(OpenBabel::OBMol &mol, OpenMM::System *sys, int maxNumOfConformations, double* &coorsConformers)
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
  int N = 1000;  // num of random conformers from which centers are selected

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
  OpenMM::Platform& platform = OpenMM::Platform::getPlatformByName("CPU");
  OpenMM::Context context(*sys, integrator, platform);
  printf( "REMARK  GeneConformation Using OpenMM platform %s\n",
	  context.getPlatform().getName().c_str() );

  OpenMM::State state;
  std::vector<OpenMM::Vec3> position(sys->getNumParticles());
  OpenBabel::OBAlign align(mol, mols[0]);
  vector<double> energy(N,0);
  for (int k = 0; k < N; k++)
  {
    for(int i = 0; i < sys->getNumParticles(); i++)
    {
      position[i] = OpenMM::Vec3(mols[k].GetCoordinates()[i*3+0]*OpenMM::NmPerAngstrom,
				 mols[k].GetCoordinates()[i*3+1]*OpenMM::NmPerAngstrom,
				 mols[k].GetCoordinates()[i*3+2]*OpenMM::NmPerAngstrom);
    }
    context.setPositions(position);
    minimizer.minimize(context, 0.001, 100);
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
  
  // get the sorted index for conformers based on its energy
  vector<size_t> idx;
  idx = sort_index<double>(energy);

  // clustering using nearst neighbour
  vector<size_t> idxCenter;
  double cutOff = 1.5;
  idxCenter.push_back(idx[0]);  
  for(int i = 1; i < N; i++)
  {
    align.SetRefMol(mols[idx[i]]);
    bool ifNewCenter = true;
    for(unsigned int j = 0; j < idxCenter.size(); j++)
    {     
      align.SetTargetMol(mols[idxCenter[j]]);
      align.Align();
      if(align.GetRMSD() <= cutOff) {ifNewCenter = false; break;}
    }
    if (ifNewCenter)
    {
      idxCenter.push_back(idx[i]);
    }
    // maxNumOfConformations
    if (idxCenter.size() == maxNumOfConformations)
    {
      break;
    }
  }
  
  // copy out the conformers
  coorsConformers = new double[idxCenter.size()*sys->getNumParticles()*3];
  for(int i = 0; i < idxCenter.size(); i++)
  {
    memcpy(&coorsConformers[i*sys->getNumParticles()*3],
	   mols[idxCenter[i]].GetCoordinates(),
	   sizeof(double)*sys->getNumParticles()*3);
  }
  return idxCenter.size();  
}
