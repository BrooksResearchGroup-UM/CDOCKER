/* @(#)AddGridForcesToOpenMMSystem.h
 */

#ifndef _ADDGRIDFORCESTOOPENMMSYSTEM_H
#define _ADDGRIDFORCESTOOPENMMSYSTEM_H 1

#include "OpenMM.h"
#include <vector>

void AddGridForcesToOpenMMSystem(int gridDimX, int gridDimY, int gridDimZ,
				 double gridMinX, double gridMinY, double gridMinZ,
				 double gridMaxX, double gridMaxY, double gridMaxZ,
				 int numOfVdwGridsUsed, float* vdwGridsUsed,
				 std::vector<int> &idxOfVdwUsed,
				 std::vector<std::vector<int> > &idxOfAtomVdwRadius,
				 float* elecGridValues,
				 OpenMM::System *sys
				 );

#endif /* _ADDGRIDFORCESTOOPENMMSYSTEM_H */

