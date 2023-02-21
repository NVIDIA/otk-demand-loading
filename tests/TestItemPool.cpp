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

#include "Memory/Allocators.h"
#include "Memory/ItemPool.h"

#include <gtest/gtest.h>

#include <vector>

using namespace demandLoading;

class IntPool : public ItemPool<int, PinnedAllocator>
{
  public:
    IntPool()
        : ItemPool<int, PinnedAllocator>( PinnedAllocator() )
    {
    }
};


class TestItemPool : public testing::Test
{
    void SetUp() { cudaFree( nullptr ); }
};

TEST_F( TestItemPool, Unused )
{
    IntPool pool;
    EXPECT_EQ( 0U, pool.size() );
}

TEST_F( TestItemPool, AllocateAndFree )
{
    IntPool pool;
    int*    item = pool.allocate();
    EXPECT_EQ( 1U, pool.size() );

    pool.free( item );
    EXPECT_EQ( 0U, pool.size() );
}

TEST_F( TestItemPool, ReuseFreedItem )
{
    // Verify that freed items are reused.  (This test is implementation specific.)
    IntPool pool;
    int*    item1 = pool.allocate();
    pool.free( item1 );
    int* item2 = pool.allocate();
    EXPECT_EQ( item1, item2 );
    pool.free( item2 );
}
