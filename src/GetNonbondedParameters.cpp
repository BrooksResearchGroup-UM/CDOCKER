#include <iostream>
#include "OpenMM.h"
#include "GetNonbondedParameters.h"

void GetNonbondedParameters(OpenMM::System *sys,
			    float *charge, float *epsilon, float *vdwRadii)
{
  // find out the index corresponding to the nonbonded force
  OpenMM::NonbondedForce *nonbondedForce;
  for (int i = 0; i < sys->getNumForces(); i++)
  {
    nonbondedForce= dynamic_cast<OpenMM::NonbondedForce*>(&(sys->getForce(i)));
    if (nonbondedForce != NULL)
    {
      break;
    }
  }

  // nonbondedForce->setCutoffDistance(1.2);
  // std::cout << "method: " << nonbondedForce->getNonbondedMethod() << std::endl;
  nonbondedForce->setNonbondedMethod(nonbondedForce->NoCutoff);
  //std::cout << "method: " << nonbondedForce->getNonbondedMethod() << std::endl;
  
  // get the charge, sigma, epsilon for each particle
  double tmpcharge, tmpsigma, tmpepsilon;
  float sigma[sys->getNumParticles()];
  for (int i = 0; i < sys->getNumParticles(); i++)
  {
    nonbondedForce->getParticleParameters(i, tmpcharge, tmpsigma, tmpepsilon);
    vdwRadii[i] = tmpsigma * OpenMM::VdwRadiusPerSigma * OpenMM::AngstromsPerNm;
    epsilon[i] = tmpepsilon * OpenMM::KcalPerKJ;
    charge[i] = tmpcharge;

  }
};
