// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/DemandLoading/DeviceContext.h>
#include <OptiXToolkit/DemandLoading/Texture2D.h>
#include <OptiXToolkit/DemandLoading/Texture2DCubic.h>

#include "TestSparseTexture.h"

using namespace demandLoading;

__global__ static void cubicTextureDrawKernel( demandLoading::DeviceContext context,
                                               unsigned int                 textureId,
                                               unsigned int                 conservativeFilter,
                                               unsigned int                 filterMode,
                                               float4*                      output,
                                               int                          width,
                                               int                          height,
                                               float2                       ddx,
                                               float2                       ddy )
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if( x >= width || y >= height )
        return;

    bool resident = true;
    float s = (x + 0.5f) / width;
    float t = (y + 0.5f) / height;
    float4 color;
    float4 dresultds, dresultdt; // unused
    resident = textureUdim<float4>( context, textureId, s, t, ddx, ddy, conservativeFilter, filterMode, &color,
                                    &dresultds, &dresultdt );

    output[y * width + x] = resident ? color : make_float4( 1.f, 0.f, 1.f, 0.f );
}

__host__ void launchCubicTextureDrawKernel( CUstream                      stream,
                                            demandLoading::DeviceContext& context,
                                            unsigned int                  textureId,
                                            unsigned int                  conservativeFilter,
                                            unsigned int                  filterMode,
                                            float4*                       output,
                                            int                           width,
                                            int                           height,
                                            float2                        ddx,
                                            float2                        ddy )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    cubicTextureDrawKernel<<<dimGrid, dimBlock, 0U, stream>>>( context, textureId, conservativeFilter, filterMode,
                                                               output, width, height, ddx, ddy );
    OTK_ERROR_CHECK( cudaGetLastError() );
}


__device__ __forceinline__ float mix( float a, float b, float x )
{
    return (1.0f-x)*a + x*b;
}

__global__ static void cubicTextureSubimageDrawKernel( demandLoading::DeviceContext context,
                                                       unsigned int                 textureId,
                                                       unsigned int                 conservativeFilter,
                                                       unsigned int                 filterMode,
                                                       float4*                      image,
                                                       float4*                      drdsImage,
                                                       int                          width,
                                                       int                          height,
                                                       float2                       uv00,
                                                       float2                       uv11,
                                                       float2                       ddx,
                                                       float2                       ddy )
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if( i >= width || j >= height )
        return;

    float x = ( i + 0.5f ) / width;
    float y = ( j + 0.5f ) / height;
    float s = mix( uv00.x, uv11.x, x );
    float t = mix( uv00.y, uv11.y, y );

    float4 val, drds, drdt;
    textureUdim<float4>( context, textureId, s, t, ddx, ddy, conservativeFilter, filterMode, &val, &drds, &drdt );

    int pixelId = j * width + i;
    image[pixelId] = val;
    drdsImage[pixelId] = float4{drds.x, drdt.x, 0.0f, 0.0f};
}

__host__ void launchCubicTextureSubimageDrawKernel( CUstream                      stream,
                                                    demandLoading::DeviceContext& context,
                                                    unsigned int                  textureId,
                                                    unsigned int                  conservativeFilter,
                                                    unsigned int                  filterMode,
                                                    float4*                       image,
                                                    float4*                       drdsImage,
                                                    int                           width,
                                                    int                           height,
                                                    float2                        uv00,
                                                    float2                        uv11,
                                                    float2                        ddx,
                                                    float2                        ddy )
{
    dim3 dimBlock( 16, 16 );
    dim3 dimGrid( ( width + dimBlock.x - 1 ) / dimBlock.x, ( height + dimBlock.y - 1 ) / dimBlock.y );
    cubicTextureSubimageDrawKernel<<<dimGrid, dimBlock, 0U, stream>>>( context, textureId, conservativeFilter, filterMode,
                                                                       image, drdsImage, width, height, uv00, uv11, ddx, ddy );
    OTK_ERROR_CHECK( cudaGetLastError() );
}
