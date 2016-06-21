/* @(#)GeneRandomConformations.h
 */

#ifndef _GENERANDOMCONFORMATIONS_H
#define _GENERANDOMCONFORMATIONS_H 1

#include <openbabel/mol.h>
#include "OpenMM.h"

int GeneRandomConformations(OpenBabel::OBMol &mol, OpenMM::System *sys, int maxNumOfConformations, double* &coorsConformers);

#endif /* _GENERANDOMCONFORMATIONS_H */

