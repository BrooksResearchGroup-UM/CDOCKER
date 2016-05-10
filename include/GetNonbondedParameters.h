/* @(#)GetNonbondedParameters.h
 */

#include "OpenMM.h"

#ifndef _GETNONBONDEDPARAMETERS_H
#define _GETNONBONDEDPARAMETERS_H 1

void GetNonbondedParameters(OpenMM::System *sys, float *charge, float *epsilon, float *vdwRadii);

#endif /* _GETNONBONDEDPARAMETERS_H */

