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

#include <OptiXToolkit/ImageSource/TiledImageSource.h>

#include "MockImageSource.h"

#include <OptiXToolkit/Memory/BitCast.h>

#include <vector_functions.h>

#include <gmock/gmock.h>

#include <algorithm>

using namespace testing;

namespace {

using TiledImageSourcePtr = std::shared_ptr<imageSource::TiledImageSource>;

class TestTiledImageSource : public Test
{
  public:
    ~TestTiledImageSource() override = default;

  protected:
    void SetUp() override;

    otk::testing::MockImageSourcePtr m_baseImage{ std::make_shared<otk::testing::MockImageSource>() };
    imageSource::TextureInfo         m_baseInfo{};
    TiledImageSourcePtr              m_tiledImage;
    CUstream                         m_stream{ otk::bit_cast<CUstream>( 0xdeadbeefULL ) };
};

void TestTiledImageSource::SetUp()
{
    m_baseInfo.width        = 1920;
    m_baseInfo.height       = 1080;
    m_baseInfo.format       = CU_AD_FORMAT_UNSIGNED_INT8;
    m_baseInfo.numChannels  = 3;
    m_baseInfo.numMipLevels = 1;
    m_baseInfo.isValid      = true;
    m_baseInfo.isTiled      = false;
    m_tiledImage            = std::make_shared<imageSource::TiledImageSource>( m_baseImage );
}

}  // namespace

TEST_F( TestTiledImageSource, create )
{
    EXPECT_TRUE( m_tiledImage );
}

TEST_F( TestTiledImageSource, closeResetsInfo )
{
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, close() );
    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );

    m_tiledImage->close();
    const imageSource::TextureInfo closedInfo = m_tiledImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_NE( closedInfo, m_baseInfo );
    EXPECT_FALSE( closedInfo.isValid );
    EXPECT_NE( closedInfo, info );
}

TEST_F( TestTiledImageSource, openReturnsTiledInfo )
{
    imageSource::TextureInfo tiledInfo{ m_baseInfo };
    tiledInfo.isTiled = true;
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, getInfo() ).Times( 0 );

    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    EXPECT_EQ( tiledInfo, info );
    EXPECT_EQ( tiledInfo, result );
}

TEST_F( TestTiledImageSource, openGetsTiledInfoOnNullPtr )
{
    imageSource::TextureInfo tiledInfo{ m_baseInfo };
    tiledInfo.isTiled = true;
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, getInfo() ).Times( 0 );

    m_tiledImage->open( nullptr );
    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    EXPECT_EQ( tiledInfo, result );
}

TEST_F( TestTiledImageSource, getInfoWithoutOpenIsInvalid )
{
    const imageSource::TextureInfo result = m_tiledImage->getInfo();

    ASSERT_FALSE( result.isValid );
}

TEST_F( TestTiledImageSource, readTileSourcesDataFromReadMipLevel )
{
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    const unsigned int mipLevel{ 0 };
    const unsigned int mipLevelWidth{ m_baseInfo.width };
    const unsigned int mipLevelHeight{ m_baseInfo.height };
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( m_baseInfo.format ) * m_baseInfo.numChannels };
    const auto fillMipLevel = [=]( char* dest, unsigned int /*mipLevel*/, unsigned int expectedWidth,
                                   unsigned int expectedHeight, CUstream /*stream*/ ) {
        for( unsigned int y = 0; y < expectedHeight; ++y )
        {
            for( unsigned int x = 0; x < expectedWidth; ++x )
            {
                const char val = static_cast<char>( ( x + y ) % 256 );
                std::fill( &dest[0], &dest[pixelSizeInBytes], val );
                dest += pixelSizeInBytes;
            }
        }
    };
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), mipLevel, mipLevelWidth, mipLevelHeight, m_stream ) )
        .WillOnce( DoAll( fillMipLevel, Return( true ) ) );
    imageSource::TextureInfo info{};
    m_tiledImage->open( &info );
    ASSERT_TRUE( info.isValid );

    const imageSource::Tile tile{ 0, 0, 64, 64 };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );
    ASSERT_TRUE( m_tiledImage->readTile( dest.data(), mipLevel, tile, m_stream ) );

    for( unsigned int y = 0; y < 64; ++y )
    {
        for( unsigned int x = 0; x < 64; ++x )
        {
            for( unsigned int c = 0; c < pixelSizeInBytes; ++c )
            {
                EXPECT_EQ( static_cast<char>( ( x + y ) % 256 ), dest[y * tile.width * pixelSizeInBytes + x * pixelSizeInBytes + c] );
            }
        }
    }
}

TEST_F( TestTiledImageSource, readMipTailReadsMipLevels )
{
    imageSource::TextureInfo baseMipInfo{ m_baseInfo };
    baseMipInfo.width        = 16;
    baseMipInfo.height       = 16;
    baseMipInfo.numMipLevels = 5;
    EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( baseMipInfo ) );
    const unsigned int mipTailFirstLevel{ 0 };
    const unsigned int numMipLevels{ 5 };
    std::vector<uint2> mipLevelDims;
    unsigned int       size = 1;
    const unsigned int pixelSizeInBytes{ imageSource::getBytesPerChannel( baseMipInfo.format ) * baseMipInfo.numChannels };
    for( unsigned int i = 0; i < numMipLevels; ++i )
    {
        mipLevelDims.push_back( make_uint2( size, size ) );
        EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), i, size, size, m_stream ) ).WillOnce( Return( true ) );
        size *= 2;
    }
    m_tiledImage->open( nullptr );

    std::vector<char> dest;
    dest.resize( getTextureSizeInBytes( baseMipInfo ) );
    EXPECT_TRUE( m_tiledImage->readMipTail( dest.data(), mipTailFirstLevel, numMipLevels, mipLevelDims.data(),
                                            pixelSizeInBytes, m_stream ) );
}

TEST_F( TestTiledImageSource, tracksTileReadCount )
{
    EXPECT_CALL( *m_baseImage, open( _ ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
    EXPECT_CALL( *m_baseImage, readMipLevel( NotNull(), 0, m_baseInfo.width, m_baseInfo.height, _ ) ).WillOnce( Return( true ) );
    m_tiledImage->open( nullptr );
    const imageSource::Tile tile{ 0, 0, 64, 64 };
    const imageSource::Tile tile2{ 1, 0, 64, 64 };
    std::vector<char>       dest;
    dest.resize( tile.width * tile.height * 4 );
    m_tiledImage->readTile( dest.data(), 0, tile, m_stream );
    m_tiledImage->readTile( dest.data(), 0, tile2, m_stream );

    EXPECT_EQ( 2ULL, m_tiledImage->getNumTilesRead() );
}

namespace {

class TestTiledImageSourcePassThrough : public TestTiledImageSource
{
  public:
    ~TestTiledImageSourcePassThrough() override = default;

  protected:
    void SetUp() override
    {
        TestTiledImageSource::SetUp();
        m_baseInfo.isTiled = true;
        EXPECT_CALL( *m_baseImage, open( NotNull() ) ).WillOnce( SetArgPointee<0>( m_baseInfo ) );
        m_tiledImage->open( nullptr );
    }
};

}  // namespace

TEST_F( TestTiledImageSourcePassThrough, open )
{
}

TEST_F( TestTiledImageSourcePassThrough, close )
{
    EXPECT_CALL( *m_baseImage, close() );

    m_tiledImage->close();
}

TEST_F( TestTiledImageSourcePassThrough, getInfo )
{
    EXPECT_CALL( *m_baseImage, getInfo() ).WillOnce( ReturnRef( m_baseInfo ) );

    const imageSource::TextureInfo info = m_tiledImage->getInfo();

    EXPECT_TRUE( info.isValid );
    EXPECT_TRUE( info.isTiled );
}

TEST_F( TestTiledImageSourcePassThrough, readTile )
{
    EXPECT_CALL( *m_baseImage, readTile( NotNull(), 1, imageSource::Tile{ 2, 3, 16, 16 }, m_stream ) ).WillOnce( Return( true ) );

    char buffer{};
    EXPECT_TRUE( m_tiledImage->readTile( &buffer, 1, { 2, 3, 16, 16 }, m_stream ) );
}

TEST_F( TestTiledImageSourcePassThrough, readMipTail )
{
    char        buffer{};
    const uint2 dims{};
    EXPECT_CALL( *m_baseImage, readMipTail( &buffer, 1, 2, &dims, 4, m_stream ) ).WillOnce( Return( true ) );

    EXPECT_TRUE( m_tiledImage->readMipTail( &buffer, 1, 2, &dims, 4, m_stream ) );
}

TEST_F( TestTiledImageSourcePassThrough, getNumTilesRead )
{
    EXPECT_CALL( *m_baseImage, getNumTilesRead() ).WillOnce( Return( 13 ) );

    EXPECT_EQ( 13, m_tiledImage->getNumTilesRead() );
}
