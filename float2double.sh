#!/bin/bash

# Created by Xinqiang Ding (xqding@umich.edu)
# at 2016/06/02 20:59:20
cd ./include
for i in `ls`;do
    echo $i
    sed -i s/float/double/g $i
    sed -i s/cufftComplex/cufftDoubleComplex/g $i
    sed -i s/cufftReal/cufftDoubleReal/g $i
    sed -i s/CUFFT_R2C/CUFFT_D2Z/g $i
    sed -i s/CUFFT_C2R/CUFFT_Z2D/g $i
    sed -i s/CUFFT_C2C/CUFFT_Z2Z/g $i
    sed -i s/cufftExecR2C/cufftExecD2Z/g $i
    sed -i s/cufftExecC2R/cufftExecZ2D/g $i
    sed -i s/cufftExecC2C/cufftExecZ2Z/g $i
done

cd ../src
for i in `ls`;do
    echo $i
    sed -i s/float/double/g $i
    sed -i s/cufftComplex/cufftDoubleComplex/g $i
    sed -i s/cufftReal/cufftDoubleReal/g $i
    sed -i s/CUFFT_R2C/CUFFT_D2Z/g $i
    sed -i s/CUFFT_C2R/CUFFT_Z2D/g $i
    sed -i s/CUFFT_C2C/CUFFT_Z2Z/g $i
    sed -i s/cufftExecR2C/cufftExecD2Z/g $i
    sed -i s/cufftExecC2R/cufftExecZ2D/g $i
    sed -i s/cufftExecC2C/cufftExecZ2Z/g $i
done

