#include "QuaternionMultiply.h"

void QuaternionMultiply(const float* const lq,
			const float* const rq,
			float* const result)
{
  result[0] = lq[0]*rq[0] - lq[1]*rq[1] - lq[2]*rq[2] - lq[3]*rq[3];
  result[1] = lq[0]*rq[1] + lq[1]*rq[0] + lq[2]*rq[3] - lq[3]*rq[2];
  result[2] = lq[0]*rq[2] - lq[1]*rq[3] + lq[2]*rq[0] + lq[3]*rq[1];
  result[3] = lq[0]*rq[3] + lq[1]*rq[2] - lq[2]*rq[1] + lq[3]*rq[0];
};
