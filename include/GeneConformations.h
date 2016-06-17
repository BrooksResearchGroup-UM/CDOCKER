/* @(#)GeneConformations.h
 */

#ifndef _GENECONFORMATIONS_H
#define _GENECONFORMATIONS_H 1

#include <vector>

template <typename T>
std::vector<size_t> sort_index(const std::vector<T> &v)
{
  // initialize original index locations
  std::vector<size_t> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
  
  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
  return idx;
};

// return num of conformer centers
int GeneConformations(OpenBabel::OBMol &mol, OpenMM::System *sys, int maxNumOfConformations, double* &coorsConformers);

#endif /* _GENECONFORMATIONS_H */

