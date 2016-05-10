#include "Rotate.h"
#include "QuaternionMultiply.h"

void Rotate(const float* const quaternion,
	    const float* const icoor, float* const fcoor)
{
  float quaternionConj[4];
  quaternionConj[0] = quaternion[0];
  quaternionConj[1] = -quaternion[1];
  quaternionConj[2] = -quaternion[2];
  quaternionConj[3] = -quaternion[3];

  float qicoor[4];
  qicoor[0] = 0;
  qicoor[1] = icoor[0];
  qicoor[2] = icoor[1];
  qicoor[3] = icoor[2];

  float tmp1[4];
  float tmp2[4];
  QuaternionMultiply(quaternion, qicoor, tmp1);
  QuaternionMultiply(tmp1, quaternionConj, tmp2);

  fcoor[0] = tmp2[1];
  fcoor[1] = tmp2[2];
  fcoor[2] = tmp2[3];
};

