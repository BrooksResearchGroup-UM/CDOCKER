#include "AddCustomNonbondedForceToOpenMMSystem.h"

#include <set>
#include <utility>
#include <iostream>
#include "OpenMM.h"

void AddCustomNonbondedForceToOpenMMSystem(OpenMM::System *bothOmmSys)
{
  // The function remove the default NonbondedForce from the system and add
  // a CustomNonbondedForce. The CustomNonbondedForce has one interaction group
  // in which one set is all the atoms with nonzero mass and the other set is
  // all the atoms in the system. In OpenMM the atoms with zero masses is fixed,
  // so we don't want to calcuate the nonbonded force between these fixed atoms.

  // get index for all atoms and nonzero mass atoms
  int numAtoms = bothOmmSys->getNumParticles();
  std::set<int> allAtoms;
  std::set<int> nonZeroMassAtoms;
  for (int i = 0; i < numAtoms; i++)
  {
    allAtoms.insert(i);
    if (bothOmmSys->getParticleMass(i) > 0)
    {
      nonZeroMassAtoms.insert(i);
    }
  }

  // get paired atom index which are bonded
  std::vector<std::pair<int, int>> bonds;
  OpenMM::HarmonicBondForce *bondForce;
  for(int i = 0; i < bothOmmSys->getNumForces(); i++)
  {
    bondForce = dynamic_cast<OpenMM::HarmonicBondForce*>(&(bothOmmSys->getForce(i)));
    if (bondForce != NULL)
    {
      break;
    }
  }

  int idx1, idx2;
  double length, k;
  for(int i = 0; i < bondForce->getNumBonds(); i++)
  {
    bondForce->getBondParameters(i, idx1, idx2, length, k);
    bonds.push_back(std::make_pair(idx1, idx2));
  }
  
  // find out the index corresponding to the nonbonded force
  OpenMM::NonbondedForce *nonbondedForce;
  int idxOfDefaultNonbondedForce;
  for (int i = 0; i < bothOmmSys->getNumForces(); i++)
  {
    nonbondedForce= dynamic_cast<OpenMM::NonbondedForce*>(&(bothOmmSys->getForce(i)));
    if (nonbondedForce != NULL)
    {
      idxOfDefaultNonbondedForce = i;
      break;
    }
  }

  // add CustomNonbondedForce for vdw
  OpenMM::CustomNonbondedForce* vdwForce =
    new OpenMM::CustomNonbondedForce("4*epsilon*((sigma/r)^12-(sigma/r)^6); sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)");
  vdwForce->addPerParticleParameter("sigma");
  vdwForce->addPerParticleParameter("epsilon");

  std::vector<double> para(2);
  double tmpcharge, tmpsigma, tmpepsilon;
  for (int i = 0; i < numAtoms; i++)
  {
    nonbondedForce->getParticleParameters(i, tmpcharge, tmpsigma, tmpepsilon);
    para[0] = tmpsigma;
    para[1] = tmpepsilon;
    vdwForce->addParticle(para);
  }
  vdwForce->addInteractionGroup(nonZeroMassAtoms, allAtoms);
  vdwForce->setForceGroup(20);
  vdwForce->createExclusionsFromBonds(bonds, 2);
  vdwForce->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffNonPeriodic);
  vdwForce->setCutoffDistance(1.4);
  vdwForce->setUseSwitchingFunction(true);
  vdwForce->setSwitchingDistance(1.2);
  // vdwForce->setNonbondedMethod(OpenMM::CustomNonbondedForce::NoCutoff);
  
  bothOmmSys->addForce(vdwForce);

  // add CustomNonbondedForce for electrostatic
  // note the formula for calculating electrostatic energy
  OpenMM::CustomNonbondedForce* elecForce =
    new OpenMM::CustomNonbondedForce("1/(4*pi*epsilon0*3)*charge1*charge2/(r^2);"); 
  elecForce->addGlobalParameter("pi", 3.1415926535897932385);
  // elecForce->addGlobalParameter("epsilon0", 0.00057276576);
  elecForce->addGlobalParameter("epsilon0", 0.0057276576);
  elecForce->addPerParticleParameter("charge");
  std::vector<double> chargePara(1);
  for (int i = 0; i < numAtoms; i++)
  {
    nonbondedForce->getParticleParameters(i, tmpcharge, tmpsigma, tmpepsilon);
    chargePara[0] = tmpcharge;
    elecForce->addParticle(chargePara);
  }
  elecForce->addInteractionGroup(nonZeroMassAtoms, allAtoms);
  elecForce->setForceGroup(21);
  elecForce->createExclusionsFromBonds(bonds, 2);
  elecForce->setNonbondedMethod(OpenMM::CustomNonbondedForce::CutoffNonPeriodic);
  elecForce->setCutoffDistance(1.4);
  elecForce->setUseSwitchingFunction(true);
  elecForce->setSwitchingDistance(1.2);
  // elecForce->setNonbondedMethod(OpenMM::CustomNonbondedForce::NoCutoff);
  bothOmmSys->addForce(elecForce);
  // remove default nonbonded force
  bothOmmSys->removeForce(idxOfDefaultNonbondedForce);  
}
