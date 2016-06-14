#include "AddGridForcesToOpenMMSystem.h"

#include "GetNonbondedParameters.h"
#include "OpenMM.h"
#include <vector>
#include <string>

void AddGridForcesToOpenMMSystem(int gridDimX, int gridDimY, int gridDimZ,
				 double gridMinX, double gridMinY, double gridMinZ,
				 double gridMaxX, double gridMaxY, double gridMaxZ,
				 int numOfVdwGridsUsed, float* vdwGridsUsed,
				 std::vector<int> &idxOfVdwUsed,
				 std::vector< std::vector<int> > &idxOfAtomVdwRadius,
				 float* elecGridValues,
				 OpenMM::System *sys
				 )
{
  // get nonbonded interaction parameters from system
  int nAtom = sys->getNumParticles();
  float atomCharges[nAtom];
  float atomEpsilons[nAtom];
  float atomRadii[nAtom];
  GetNonbondedParameters(sys, atomCharges, atomEpsilons, atomRadii);

  // build customize force for electrostatic energy
  // cope the grid values. Note that the fastest index in CHARMM and OpenMM are different.
  std::vector <double> tmpGridValues(gridDimX*gridDimY*gridDimZ, 0);
  int idxForGridOpenMM;
  int idxForGridCHARMM;
  for(int i = 0; i < gridDimX; i++)
  {
    for(int j = 0; j < gridDimY; j++)
    {
      for(int k = 0; k < gridDimZ; k++)
      {
	idxForGridCHARMM = (i * gridDimY + j) * gridDimZ + k; // in CHARMM grid, the last index changes fastest
	idxForGridOpenMM = (k * gridDimY + j) * gridDimX + i; // in OpenMM grid, the first index changes fastest
	tmpGridValues[idxForGridOpenMM] = elecGridValues[idxForGridCHARMM];
      }
    }
  }
  
  OpenMM::Continuous3DFunction *elecGridFunction =
    new OpenMM::Continuous3DFunction(gridDimX, gridDimY, gridDimZ,
  				     tmpGridValues,
  				     gridMinX*OpenMM::NmPerAngstrom,
  				     gridMaxX*OpenMM::NmPerAngstrom,
  				     gridMinY*OpenMM::NmPerAngstrom,
  				     gridMaxY*OpenMM::NmPerAngstrom,
  				     gridMinZ*OpenMM::NmPerAngstrom,
  				     gridMaxZ*OpenMM::NmPerAngstrom);
  
  OpenMM::CustomCompoundBondForce *elecGridPotential =
    new OpenMM::CustomCompoundBondForce(1, "elecGrid(x1,y1,z1) * q * KJPerKcal");
  elecGridPotential->setForceGroup(10);
  int idxElecGrid = sys->addForce(elecGridPotential);
  elecGridPotential->addTabulatedFunction("elecGrid", elecGridFunction);
  elecGridPotential->addGlobalParameter("KJPerKcal", OpenMM::KJPerKcal);
  elecGridPotential->addPerBondParameter("q");
  std::vector<int> idxParticle(1,0);
  std::vector<double> parameter(1,0);  
  for (int i = 0; i < sys->getNumParticles(); i++)
  {
    idxParticle[0] = i;
    parameter[0] = atomCharges[i];
    elecGridPotential->addBond(idxParticle, parameter);
  }

  // build customized force for van der Waals interaction
  OpenMM::Continuous3DFunction *vdwGridFunctions[numOfVdwGridsUsed];
  OpenMM::CustomCompoundBondForce *vdwGridPotentials[numOfVdwGridsUsed];
  std::string formula;
  for(int l = 0; l < numOfVdwGridsUsed; l++)
  {
    // copy over grid values
    for(int i = 0; i < gridDimX; i++)
    {
      for(int j = 0; j < gridDimY; j++)
      {
	for(int k = 0; k < gridDimZ; k++)
	{
	  idxForGridCHARMM = (i * gridDimY + j) * gridDimZ + k; // in CHARMM grid, the last index changes fastest
	  idxForGridOpenMM = (k * gridDimY + j) * gridDimX + i; // in OpenMM grid, the first index changes fastest
	  tmpGridValues[idxForGridOpenMM] = vdwGridsUsed[l*gridDimX*gridDimY*gridDimZ + idxForGridCHARMM];
	}
      }
    }
    
    vdwGridFunctions[l] =
      new OpenMM::Continuous3DFunction(gridDimX, gridDimY, gridDimZ,
  				       tmpGridValues,
  				       gridMinX * OpenMM::NmPerAngstrom,
  				       gridMaxX * OpenMM::NmPerAngstrom,
  				       gridMinY * OpenMM::NmPerAngstrom,
  				       gridMaxY * OpenMM::NmPerAngstrom,
  				       gridMinZ * OpenMM::NmPerAngstrom,
  				       gridMaxZ * OpenMM::NmPerAngstrom);
    formula = "vdwGrid";
    formula += std::to_string(l);
    formula += "(x1,y1,z1) * sqrt(epsilon) * KJPerKcal";
    
    vdwGridPotentials[l] = new OpenMM::CustomCompoundBondForce(1, formula);
    vdwGridPotentials[l]->setForceGroup(11);
    sys->addForce(vdwGridPotentials[l]);
    formula = "vdwGrid";
    formula += std::to_string(l);
	
    vdwGridPotentials[l]->addTabulatedFunction(formula, vdwGridFunctions[l]);
    vdwGridPotentials[l]->addGlobalParameter("KJPerKcal", OpenMM::KJPerKcal);
    vdwGridPotentials[l]->addPerBondParameter("epsilon");

    for (int i = 0; i < idxOfAtomVdwRadius[idxOfVdwUsed[l]].size(); i++)
    {
      int idx = idxOfAtomVdwRadius[idxOfVdwUsed[l]][i];
      idxParticle[0] = idx;
      parameter[0] = atomEpsilons[idx];
      vdwGridPotentials[l]->addBond(idxParticle, parameter);
    }
  }
}
