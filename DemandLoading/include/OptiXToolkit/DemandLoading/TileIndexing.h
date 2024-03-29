//
// Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#pragma once

#include <OptiXToolkit/DemandLoading/TextureSampler.h>

#ifndef __CUDACC_RTC__
#include <cuda.h>
#include <texture_types.h>
#else
enum CUaddress_mode {
    CU_TR_ADDRESS_MODE_WRAP   = 0,
    CU_TR_ADDRESS_MODE_CLAMP  = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
    CU_TR_ADDRESS_MODE_BORDER = 3 
};
#endif

#ifndef __CUDACC__
#include <algorithm>
#include <cmath>
#endif

namespace demandLoading {

#define HD_INLINE __host__ __device__ static __forceinline__
#define D_INLINE __device__ static __forceinline__

// clang-format off
#ifdef __CUDACC__
HD_INLINE float maxf( float x, float y ) { return ::fmaxf( x, y ); }
HD_INLINE float minf( float x, float y ) { return ::fminf( x, y ); }
HD_INLINE unsigned int uimax( unsigned int a, unsigned int b ) { return ( a > b ) ? a : b; }
#else
HD_INLINE float maxf( float x, float y ) { return std::max( x, y ); }
HD_INLINE float minf( float x, float y ) { return std::min( x, y ); }
HD_INLINE unsigned int uimax( unsigned int a, unsigned int b ) { return std::max( a, b ); }
#endif
HD_INLINE float clampf( float f, float a, float b ) { return maxf( a, minf( f, b ) ); }
// clang-format on

HD_INLINE unsigned int calculateLevelDim( unsigned int mipLevel, unsigned int textureDim ) 
{ 
    return uimax( textureDim >> mipLevel, 1U ); 
}

HD_INLINE unsigned int getLevelDimInTiles( unsigned int textureDim, unsigned int mipLevel, unsigned int tileDim )
{
    return ( calculateLevelDim( mipLevel, textureDim ) + tileDim - 1 ) / tileDim;
}

HD_INLINE unsigned int calculateNumTilesInLevel( unsigned int levelWidthInTiles, unsigned int levelHeightInTiles )
{
    return levelWidthInTiles * levelHeightInTiles;
}

HD_INLINE void getTileCoordsFromPageOffset( int pageOffsetInLevel, int levelWidthInTiles, unsigned int& tileX, unsigned int& tileY )
{
    tileX = pageOffsetInLevel % levelWidthInTiles;
    tileY = pageOffsetInLevel / levelWidthInTiles;
}

HD_INLINE float wrapTexCoord( float x, CUaddress_mode addressMode )
{
    const float firstFloatLessThanOne = 0.999999940395355224609375f;
    if( addressMode == CU_TR_ADDRESS_MODE_WRAP )
        x = x - floorf( x );
    return clampf( x, 0.0f, firstFloatLessThanOne );
}

HD_INLINE int getPageOffsetFromTileCoords( int x, int y, int levelWidthInTiles )
{
    return y * levelWidthInTiles + x;
}

// Return the mip level and pixel coordinates of the corner of the tile associated with tileIndex
HD_INLINE void unpackTileIndex( const TextureSampler& sampler, unsigned int tileIndex, 
                                unsigned int& outMipLevel, unsigned int& outTileX, unsigned int& outTileY )
{
    for( int mipLevel = sampler.mipTailFirstLevel; mipLevel >= 0; --mipLevel )
    {
        unsigned int nextMipLevelStart = ( mipLevel > 0 ) ? sampler.mipLevelSizes[mipLevel - 1].mipLevelStart : sampler.numPages;
        if( tileIndex < nextMipLevelStart )
        {
            const unsigned int levelWidthInTiles = sampler.mipLevelSizes[mipLevel].levelWidthInTiles;
            const unsigned int indexInLevel      = tileIndex - sampler.mipLevelSizes[mipLevel].mipLevelStart;
            outMipLevel                          = mipLevel;
            getTileCoordsFromPageOffset( indexInLevel, levelWidthInTiles, outTileX, outTileY );
            return;
        }
    }
    outMipLevel = 0;
    outTileX    = 0;
    outTileY    = 0;
}

HD_INLINE bool isMipTailIndex( unsigned int pageIndex )
{
    return pageIndex == 0; // Page 0 always contains the mip tail.
}

}  // namespace demandLoading
