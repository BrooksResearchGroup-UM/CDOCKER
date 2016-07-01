#include "RemoveNonbondedForceBetweenMasslessParticles.h"

#include "OpenMM.h"

void RemoveNonbondedForceBetweenMasslessParticles(OpenMM::System *sys)
{
  // get index for all massless particles
  int numAtoms = sys->getNumParticles();
  std::vector<int> idxMasslessAtoms;
  for (int i = 0; i < numAtoms; i++)
  {
    if (sys->getParticleMass(i) == 0)
    {
      idxMasslessAtoms.push_back(i);
    }
  }

  // find out the index corresponding to the nonbonded force
  OpenMM::NonbondedForce *nonbondedForce;
  int idxOfDefaultNonbondedForce;
  for (int i = 0; i < sys->getNumForces(); i++)
  {
    nonbondedForce= dynamic_cast<OpenMM::NonbondedForce*>(&(sys->getForce(i)));
    if (nonbondedForce != NULL)
    {
      idxOfDefaultNonbondedForce = i;
      break;
    }
  }

  for(int i = 0; i < idxMasslessAtoms.size(); i++)
  {
    for(int j = i + 1; j < idxMasslessAtoms.size(); j++)
    {
      nonbondedForce->addException(idxMasslessAtoms[i], idxMasslessAtoms[j], 0, 0, 0);
    }
  }
}
