/* @(#)GetIdxOfAtomsForVdwRadius.h
 */
#include <vector>

#ifndef _GETIDXOFATOMSFORVDWRADIUS_H
#define _GETIDXOFATOMSFORVDWRADIUS_H 1

void GetIdxOfAtomsForVdwRadius(int nAtoms, float* atomRadii,
			       int numOfVdwGrids, float *gridRadii,
			       int &numOfVdwGridsUsed, std::vector<int> &idxOfVdwUsed,
			       std::vector< std::vector<int> > &idxOfAtomVdwRadius);

#endif /* _GETIDXOFATOMSFORVDWRADIUS_H */

