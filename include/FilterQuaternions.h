/* @(#)FilterQuaternions.h
 */

#ifndef _FILTERQUATERNIONS_H
#define _FILTERQUATERNIONS_H 1

int FilterQuaternions(float* coor, int nAtom,
		      int nQuater, float* quaternions,
		      float gridLenX, float gridLenY, float gridLenZ,
		      int maxNQuaternionsUsed, float* &quaternionsUsed);

#endif /* _FILTERQUATERNIONS_H */

