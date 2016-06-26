#include <iostream>
#include <fstream>

#include "OpenMM.h"
#include "ReadCrd.h"

int main(int argc, char** argv)
{
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
  
  ligandProteinState = ligandProteinContext.getState(OpenMM::State::Energy);
  std::cout << ligandProteinState.getPotentialEnergy() * OpenMM::KcalPerKJ << std::endl;
  
  return 0;
}
