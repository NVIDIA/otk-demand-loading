//
// Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

const unsigned int DEMAND_TEXTURE_VIRTUAL_PAGE_ALIGNMENT = 32;

// clang-format off
#ifdef __CUDACC__
__host__ __device__ static __forceinline__ float maxf( float x, float y ) { return ::fmaxf( x, y ); }
__host__ __device__ static __forceinline__ float minf( float x, float y ) { return ::fminf( x, y ); }
__host__ __device__ static __forceinline__ unsigned int uimax( unsigned int a, unsigned int b ) { return ( a > b ) ? a : b; }
#else
__host__ __device__ static __forceinline__ float maxf( float x, float y ) { return std::max( x, y ); }
__host__ __device__ static __forceinline__ float minf( float x, float y ) { return std::min( x, y ); }
__host__ __device__ static __forceinline__ unsigned int uimax( unsigned int a, unsigned int b ) { return std::max( a, b ); }
#endif
__host__ __device__ static __forceinline__ float clampf( float f, float a, float b ) { return maxf( a, minf( f, b ) ); }
// clang-format on

__host__ __device__ static __forceinline__ unsigned int calculateLevelDim( unsigned int mipLevel, unsigned int textureDim )
{
    return uimax( textureDim >> mipLevel, 1U );
}

__host__ __device__ static __forceinline__ unsigned int getLevelDimInTiles( unsigned int textureDim, unsigned int mipLevel, unsigned int tileDim )
{
    return ( calculateLevelDim( mipLevel, textureDim ) + tileDim - 1 ) / tileDim;
}

__host__ __device__ static __forceinline__ unsigned int calculateNumTilesInLevel( unsigned int levelWidthInTiles,
                                                                                  unsigned int levelHeightInTiles )
{
    // Round up width and height to multiples of 8 to be consistent with hardware footprint alignment.
    levelWidthInTiles  = ( levelWidthInTiles + 7 ) & 0xfffffff8;
    levelHeightInTiles = ( levelHeightInTiles + 7 ) & 0xfffffff8;
    return levelWidthInTiles * levelHeightInTiles;
}

__host__ __device__ static __forceinline__ void getTileCoordsFromPageOffset( int           pageOffsetInLevel,
                                                                             int           levelWidthInTiles,
                                                                             unsigned int& tileX,
                                                                             unsigned int& tileY )
{
    int levelWidthInBlocks = ( levelWidthInTiles + 7 ) / 8;

    int blockNum = pageOffsetInLevel / 64;
    int xblock   = blockNum % levelWidthInBlocks;
    int yblock   = blockNum / levelWidthInBlocks;

    int blockOffset = pageOffsetInLevel % 64;
    int xoffset     = blockOffset % 8;
    int yoffset     = blockOffset / 8;

    tileX = xblock * 8 + xoffset;
    tileY = yblock * 8 + yoffset;
}

__host__ __device__ static __forceinline__ float wrapTexCoord( float x, CUaddress_mode addressMode )
{
    const float firstFloatLessThanOne = 0.999999940395355224609375f;
    return ( addressMode == CU_TR_ADDRESS_MODE_WRAP ) ? x - floorf( x ) : clampf( x, 0.0f, firstFloatLessThanOne );
}

__host__ __device__ static __forceinline__ int getPageOffsetFromTileCoords( int x, int y, int levelWidthInTiles )
{
    // Tiles are layed out in 8x8 blocks to match the output of the texture footprint instruction
    int levelWidthInBlocks = ( levelWidthInTiles + 7 ) / 8;
    return 64 * ( levelWidthInBlocks * ( y / 8 ) ) +  // Offset for full rows of blocks
           64 * ( x / 8 ) +                           // Offset for full blocks on last row
           8 * ( y % 8 ) + ( x % 8 );                 // Partial offset in last block
}

// Return the mip level and pixel coordinates of the corner of the tile associated with tileIndex
__host__ __device__ static __forceinline__ void unpackTileIndex( const TextureSampler& sampler,
                                                                 unsigned int             tileIndex,
                                                                 unsigned int&            outMipLevel,
                                                                 unsigned int&            outTileX,
                                                                 unsigned int&            outTileY )
{
    const demandLoading::TextureSampler::MipLevelSizes* mls = sampler.mipLevelSizes;
    for( int mipLevel = sampler.mipTailFirstLevel; mipLevel >= 0; --mipLevel )
    {
        unsigned int nextMipLevelStart = ( mipLevel > 0 ) ? mls[mipLevel - 1].mipLevelStart : sampler.numPages;
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

__host__ __device__ static __forceinline__ bool isMipTailIndex( unsigned int pageIndex )
{
    // Page 0 always contains the mip tail.
    return pageIndex == 0;
}

// Wrap the tile coordinate x based on the toroidal addressing scheme of the footprint instruction
__host__ __device__ static __forceinline__ int wrapFootprintTileCoord( int x, unsigned int dx, unsigned int tileX, unsigned int levelWidthInTiles )
{
    // Toroidal rotation
    if( x + dx >= 8 )
        x -= 8;

    // Fix spillover that happens on small levels sometimes
    if( x > (int)levelWidthInTiles )
        x = 0;

    // Add base tile contribution
    x += 8 * tileX;

    // Wrap the tile coord
    if( x < 0 )
        x += levelWidthInTiles;
    if( x >= static_cast<int>( levelWidthInTiles ) )
        x -= levelWidthInTiles;
    return x;
}

}  // namespace demandLoading
