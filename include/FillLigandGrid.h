/* @(#)FillLigandGrid.h
 */

#ifndef _FILLLIGANDGRID_H
#define _FILLLIGANDGRID_H 1

void FillLigandGrid(int nQuaternions,
		    int nAtoms, float *coors, float *mincoors,
		    float *charge, float *epsilon,
		    int numOfVdwGridsUsed, std::vector<int> idxOfVdwUsed,
		    std::vector< std::vector<int> > &idxOfAtomVdwRadius,
		    int xdim, int ydim, int zdim,
		    float spacing, float *ligandGridValues
		    );


#endif /* _FILLLIGANDGRID_H */

