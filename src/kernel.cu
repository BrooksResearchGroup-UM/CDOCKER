#include "kernel.h"
#include <cufft.h>

/////  GPU kernels /////
__global__ void ConjMult(cufftComplex *d_potential_F, cufftComplex *d_ligand_F, int odist, int numOfGridsUsed)
{
  int dist = odist * numOfGridsUsed;
  int idx_l = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int idx_p = idx_l % dist;
  float x = d_ligand_F[idx_l].x;
  float y = d_ligand_F[idx_l].y;
  d_ligand_F[idx_l].x = x * d_potential_F[idx_p].x + y * d_potential_F[idx_p].y;
  d_ligand_F[idx_l].y = x * d_potential_F[idx_p].y - y * d_potential_F[idx_p].x;
};

__global__ void SumGrids(cufftComplex *d_ligand_F, cufftComplex *d_ligand_sum_F, int numOfGridsUsed, int odist, int idist)
{
  int idx = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
  int idxQuaternion = idx / odist;
  int idxValue = idx % odist;
  d_ligand_sum_F[idx].x = 0;
  d_ligand_sum_F[idx].y = 0;
  for(int i = 0; i < numOfGridsUsed; i++)
  {
    d_ligand_sum_F[idx].x += d_ligand_F[ (idxQuaternion*numOfGridsUsed + i) * odist + idxValue].x;
    d_ligand_sum_F[idx].y += d_ligand_F[ (idxQuaternion*numOfGridsUsed + i) * odist + idxValue].y;
  }
  d_ligand_sum_F[idx].x = d_ligand_sum_F[idx].x / sqrt((float) idist);
  d_ligand_sum_F[idx].y = d_ligand_sum_F[idx].y / sqrt((float) idist);
};
