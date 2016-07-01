#include <iostream>
#include <fstream>

#include <openbabel/mol.h>
#include <openbabel/obconversion.h>
#include "OpenMM.h"
#include "ReadCrd.h"
#include "AddCustomNonbondedForceToOpenMMSystem.h"
#include "CalcRMSD.h"

int main(int argc, char** argv)
{
  OpenMM::Platform::loadPluginsFromDirectory(
  					     "/home/xqding/apps/openmmDev/lib/plugins");

  // read ligand molecule
  OpenBabel::OBMol ligandOBMol;
  OpenBabel::OBConversion conv(&std::cin, &std::cout);
  conv.SetInFormat("mol2");
  conv.SetOutFormat("pdb");
  conv.ReadFile(&ligandOBMol, "1G9V.lig.am1bcc.mol2");
  int nAtom = ligandOBMol.NumAtoms();

  // read openmm system
  std::ifstream sysFile;
  sysFile.open("./ligand-protein.xml", std::ifstream::in);
  OpenMM::System *bothOmmSys = new OpenMM::System();
  bothOmmSys = OpenMM::XmlSerializer::deserialize<OpenMM::System>(sysFile);
  
  // read coordinates for both ligand and protein
  float *bothCoor = 0;
  std::string bothCrdFileName = "./ligand-protein.crd";
  ReadCrd(bothOmmSys->getNumParticles(), bothCoor, bothCrdFileName);
  std::cout << bothOmmSys->getNumParticles() << std::endl;

  // build OpenMM ligand protein context
  AddCustomNonbondedForceToOpenMMSystem(bothOmmSys);
  OpenMM::LangevinIntegrator ligandProteinIntegrator(300, 5, 0.001);
  OpenMM::LocalEnergyMinimizer ligandProteinMinimizer;
  OpenMM::Context ligandProteinContext(*bothOmmSys, ligandProteinIntegrator);  
  printf( "REMARK  Build ligandProteinContext Using OpenMM platform %s\n",
  	  ligandProteinContext.getPlatform().getName().c_str() );
  OpenMM::State ligandProteinState;
  std::vector<OpenMM::Vec3> ligandProteinPosition(bothOmmSys->getNumParticles());
  for(int i = 0; i < bothOmmSys->getNumParticles(); i++)
  {
    ligandProteinPosition[i] = OpenMM::Vec3(bothCoor[i*3+0]*OpenMM::NmPerAngstrom,
					    bothCoor[i*3+1]*OpenMM::NmPerAngstrom,
					    bothCoor[i*3+2]*OpenMM::NmPerAngstrom);
  }
  ligandProteinContext.setPositions(ligandProteinPosition);  
  ligandProteinState = ligandProteinContext.getState(OpenMM::State::Energy, false, 1<<20);
  std::cout << "vdw: "<< ligandProteinState.getPotentialEnergy() * OpenMM::KcalPerKJ << std::endl;
  ligandProteinState = ligandProteinContext.getState(OpenMM::State::Energy, false, 1<<21);
  std::cout << "elec: "<< ligandProteinState.getPotentialEnergy() * OpenMM::KcalPerKJ << std::endl;
  ligandProteinState = ligandProteinContext.getState(OpenMM::State::Energy);
  std::cout << "total: "<< ligandProteinState.getPotentialEnergy() * OpenMM::KcalPerKJ << std::endl;

  OpenBabel::OBMol ligandNativeMini = ligandOBMol;
  double rmsdNativeMini = CalcRMSD(ligandOBMol, ligandNativeMini);
  std::cout << "rmsd: " << rmsdNativeMini << std::endl;
  
  ligandProteinMinimizer.minimize(ligandProteinContext, 0.001, 1000);
  double tmpCoor[ligandOBMol.NumAtoms()*3];
  ligandProteinState = ligandProteinContext.getState(OpenMM::State::Positions);
  // for(int i = 0; i < ligandOBMol.NumAtoms(); i++)
  // {
  //   tmpCoor[i*3 + 0] = ligandProteinPosition[i][0] * OpenMM::AngstromsPerNm;
  //   tmpCoor[i*3 + 1] = ligandProteinPosition[i][1] * OpenMM::AngstromsPerNm;
  //   tmpCoor[i*3 + 2] = ligandProteinPosition[i][2] * OpenMM::AngstromsPerNm;
  // }
  for(int i = 0; i < ligandOBMol.NumAtoms(); i++)
  {
    tmpCoor[i*3 + 0] = ligandProteinState.getPositions()[i][0] * OpenMM::AngstromsPerNm;
    tmpCoor[i*3 + 1] = ligandProteinState.getPositions()[i][1] * OpenMM::AngstromsPerNm;
    tmpCoor[i*3 + 2] = ligandProteinState.getPositions()[i][2] * OpenMM::AngstromsPerNm;
  }      

  ligandNativeMini.SetCoordinates(tmpCoor);
  rmsdNativeMini = CalcRMSD(ligandOBMol, ligandNativeMini);
  std::cout << "rmsd: " << rmsdNativeMini << std::endl;
  return 0;
}
