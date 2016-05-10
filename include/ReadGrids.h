/* @(#)ReadGrids.h
 */

#ifndef _READGRIDS_H
#define _READGRIDS_H 1

void ReadGrids(int &numOfGrids, int &xdim, int &ydim, int &zdim,
	       float &midx, float &midy, float &midz,
	       float &xlen, float &ylen, float &zlen,
	       float &spacing, float &restraint_k,
	       float *&raddi, float*&values,
	       char *fileName);

#endif /* _READGRIDS_H */

