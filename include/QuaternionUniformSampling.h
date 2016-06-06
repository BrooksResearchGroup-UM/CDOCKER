/* @(#)QuaternionUniformSampling.h
 */
#ifndef _QUATERNIONUNIFORMSAMPLING_H
#define _QUATERNIONUNIFORMSAMPLING_H 1

#include <random>
#include <math.h>
#include <cmath>

void QuaternionUniformSampling(std::mt19937_64& gen, float* &quaters, int nquaters);

#endif /* _QUATERNIONUNIFORMSAMPLING_H */

