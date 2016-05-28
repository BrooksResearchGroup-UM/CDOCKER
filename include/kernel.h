/* @(#)kernel.h
 */

#ifndef _KERNEL_H
#define _KERNEL_H 1

#include <cufft.h>

__global__ void ConjMult(cufftComplex *d_potential_F, cufftComplex *d_ligand_F, int odist, int numOfGridsUsed);

__global__ void SumGrids(cufftComplex *d_ligand_F, cufftComplex *d_ligand_sum_F, int numOfGridsUsed, int odist, int idist);

#endif /* _KERNEL_H */

