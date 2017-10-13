// ========================================================================== //
// The MIT License (MIT)                                                      //
//                                                                            //
// Copyright (c) 2017 Jefferson Amstutz                                       //
//                                                                            //
// Permission is hereby granted, free of charge, to any person obtaining a    //
// copy of this software and associated documentation files (the "Software"), //
// to deal in the Software without restriction, including without limitation  //
// the rights to use, copy, modify, merge, publish, distribute, sublicense,   //
// and/or sell copies of the Software, and to permit persons to whom the      //
// Software is furnished to do so, subject to the following conditions:       //
//                                                                            //
// The above copyright notice and this permission notice shall be included in //
// in all copies or substantial portions of the Software.                     //
//                                                                            //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    //
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    //
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        //
// DEALINGS IN THE SOFTWARE.                                                  //
// ========================================================================== //

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "pico_bench.h"

#include "psimd/psimd.h"

#include "../embc/simd/simd.h"

//ispc
#ifdef PSIMD_ENABLE_ISPC
#  include "mandelbrot_ispc.h"
#endif

// helper function to write the rendered image as PPM file
inline void writePPM(const std::string &fileName,
                     const int sizeX, const int sizeY,
                     const int *pixel)
{
  FILE *file = fopen(fileName.c_str(), "wb");
  fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
  unsigned char *out = (unsigned char *)alloca(3*sizeX);
  for (int y = 0; y < sizeY; y++) {
    const unsigned char *in = (const unsigned char *)&pixel[(sizeY-1-y)*sizeX];
    for (int x = 0; x < sizeX; x++) {
      out[3*x + 0] = in[4*x + 0];
      out[3*x + 1] = in[4*x + 1];
      out[3*x + 2] = in[4*x + 2];
    }
    fwrite(out, 3*sizeX, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);
}

// psimd version //////////////////////////////////////////////////////////////

namespace psimd {

using vfloat = psimd::pack<float>;
using vint   = psimd::pack<int>;
using vmask  = psimd::mask<>;

static vint programIndex(0);

inline vint mandel(const vmask &_active,
                   const vfloat &c_re,
                   const vfloat &c_im,
                   int maxIters)
{
  vfloat z_re = c_re;
  vfloat z_im = c_im;
  vint vi(0);

  for (int i = 0; i < maxIters; ++i) {
    auto active = _active && ((z_re * z_re + z_im * z_im) <= 4.f);
    if (psimd::none(active))
      break;

    vfloat new_re = z_re * z_re - z_im * z_im;
    vfloat new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;

    vi = psimd::select(active, vi + 1, vi);
  }

  return vi;
}

void mandelbrot(float x0, float y0,
                float x1, float y1,
                int width, int height, int maxIters,
                int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i += DEFAULT_WIDTH) {
      vfloat x = x0 + (i + programIndex.as<float>()) * dx;
      vfloat y = y0 + j * dy;

      auto active = x < width;

      int base_index = (j * width + i);
      auto result = mandel(active, x, y, maxIters);

      psimd::store(result, output + base_index, active);
    }
  }
}

} // ::psimd

// embree version /////////////////////////////////////////////////////////////

namespace embc {

// foreach //

template <typename SIMD_T, typename FCN_T>
inline void foreach_v(SIMD_T &v, FCN_T &&fcn)
{
  // NOTE(jda) - need to static_assert() FCN_T's signature

  for (int i = 0; i < SIMD_T::size; ++i)
    fcn(v[i], i);
}

using vfloat = embree::vfloatx;
using vint   = embree::vintx;
using vmask  = embree::vboolx;

static vint programIndex(0);

inline vint mandel(const vmask &_active,
                   const vfloat &c_re,
                   const vfloat &c_im,
                   int maxIters)
{
  vfloat z_re = c_re;
  vfloat z_im = c_im;
  vint vi(0);

  for (int i = 0; i < maxIters; ++i) {
    auto active = _active & ((z_re * z_re + z_im * z_im) <= 4.f);
    if (embree::none(active))
      break;

    vfloat new_re = z_re * z_re - z_im * z_im;
    vfloat new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;

    vi = embree::select(active, vi + 1, vi);
  }

  return vi;
}

void mandelbrot(float x0, float y0,
                float x1, float y1,
                int width, int height, int maxIters,
                int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; i += DEFAULT_WIDTH) {
      vfloat x = x0 + (i + vfloat(programIndex)) * dx;
      vfloat y = y0 + j * dy;

      auto active = x < width;

      int base_index = (j * width + i);
      auto result = mandel(active, x, y, maxIters);

      vint::store(active, output + base_index, result);
    }
  }
}

} // ::embc

// omp version ////////////////////////////////////////////////////////////////

namespace omp {

#pragma omp declare simd
inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

void mandelbrot(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  for (int j = 0; j < height; j++) {
#   pragma omp simd
    for (int i = 0; i < width; ++i) {
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}

} // ::omp

// scalar version /////////////////////////////////////////////////////////////

namespace scalar {

inline int mandel(float c_re, float c_im, int count)
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.f)
      break;

    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
  }

  return i;
}

void mandelbrot(float x0, float y0, float x1, float y1,
                int width, int height, int maxIterations,
                int output[])
{
  float dx = (x1 - x0) / width;
  float dy = (y1 - y0) / height;

  for (int j = 0; j < height; j++) {
    for (int i = 0; i < width; ++i) {
      float x = x0 + i * dx;
      float y = y0 + j * dy;

      int index = (j * width + i);
      output[index] = mandel(x, y, maxIterations);
    }
  }
}

} // ::scalar

int main()
{
  using namespace std::chrono;

  const unsigned int width  = 1200;
  const unsigned int height = 800;
  const float x0 = -2;
  const float x1 = 1;
  const float y0 = -1;
  const float y1 = 1;

  const int maxIters = 256;
  std::vector<int> buf(width*height);

  psimd::foreach(psimd::programIndex, [](int &v, int i) { v = i; });
  embc::foreach_v(embc::programIndex, [](int &v, int i) { v = i; });

  auto bencher = pico_bench::Benchmarker<milliseconds>{16, seconds{4}};

  std::cout << "starting benchmarks (results in 'ms')... " << '\n';

  // scalar run ///////////////////////////////////////////////////////////////

  std::fill(buf.begin(), buf.end(), 0);

  auto stats = bencher([&](){
    scalar::mandelbrot(x0, y0, x1, y1, width, height, maxIters, buf.data());
  });

  const float scalar_min = stats.min().count();

  std::cout << '\n' << "scalar " << stats << '\n';

  // omp run //////////////////////////////////////////////////////////////////

  std::fill(buf.begin(), buf.end(), 0);

  stats = bencher([&](){
    omp::mandelbrot(x0, y0, x1, y1, width, height, maxIters, buf.data());
  });

  const float omp_min = stats.min().count();

  std::cout << '\n' << "omp " << stats << '\n';

  // ispc run /////////////////////////////////////////////////////////////////

#ifdef PSIMD_ENABLE_ISPC
  std::fill(buf.begin(), buf.end(), 0);

  stats = bencher([&](){
    ispc::mandelbrot(x0, y0, x1, y1, width, height, maxIters, buf.data());
  });

  const float ispc_min = stats.min().count();

  std::cout << '\n' << "ispc " << stats << '\n';
#endif

  // psimd run ////////////////////////////////////////////////////////////////

  std::fill(buf.begin(), buf.end(), 0);

  stats = bencher([&](){
    psimd::mandelbrot(x0, y0, x1, y1, width, height, maxIters, buf.data());
  });

  const float psimd_min = stats.min().count();

  std::cout << '\n' << "psimd " << stats << '\n';

  // embree run ///////////////////////////////////////////////////////////////

  std::fill(buf.begin(), buf.end(), 0);

  stats = bencher([&](){
    embc::mandelbrot(x0, y0, x1, y1, width, height, maxIters, buf.data());
  });

  const float embree_min = stats.min().count();

  std::cout << '\n' << "embree " << stats << '\n';

  // conclusions //////////////////////////////////////////////////////////////

  std::cout << '\n' << "Conclusions: " << '\n';

  // scalar //

  std::cout << '\n' << "--> scalar was " << psimd_min / scalar_min
            << "x the speed of psimd" << '\n';

  std::cout << '\n' << "--> scalar was " << omp_min / scalar_min
            << "x the speed of omp" << '\n';

  std::cout << '\n' << "--> scalar was " << embree_min / scalar_min
            << "x the speed of embc" << '\n';

  // psimd //

  std::cout << '\n' << "--> psimd was " << scalar_min / psimd_min
            << "x the speed of scalar" << '\n';

  std::cout << '\n' << "--> psimd was " << omp_min / psimd_min
            << "x the speed of omp" << '\n';

  std::cout << '\n' << "--> psimd was " << embree_min / psimd_min
            << "x the speed of embc" << '\n';

  // omp //

  std::cout << '\n' << "--> omp was " << scalar_min / omp_min
            << "x the speed of scalar" << '\n';

  std::cout << '\n' << "--> omp was " << psimd_min / omp_min
            << "x the speed of psimd" << '\n';

  std::cout << '\n' << "--> omp was " << embree_min / omp_min
            << "x the speed of embc" << '\n';

  // embree //

  std::cout << '\n' << "--> embc was " << scalar_min / embree_min
            << "x the speed of scalar" << '\n';

  std::cout << '\n' << "--> embc was " << psimd_min / embree_min
            << "x the speed of psimd" << '\n';

  std::cout << '\n' << "--> embc was " << omp_min / embree_min
            << "x the speed of omp" << '\n';

  // ispc //

#ifdef PSIMD_ENABLE_ISPC
  std::cout << '\n' << "--> ispc was " << scalar_min / ispc_min
            << "x the speed of scalar" << '\n';

  std::cout << '\n' << "--> ispc was " << psimd_min / ispc_min
            << "x the speed of psimd" << '\n';

  std::cout << '\n' << "--> ispc was " << omp_min / ispc_min
            << "x the speed of omp" << '\n';

  std::cout << '\n' << "--> ispc was " << embree_min / ispc_min
            << "x the speed of embc" << '\n';
#endif

  writePPM("mandelbrot.ppm", width, height, buf.data());

  std::cout << '\n' << "wrote output image to 'mandelbrot.ppm'" << '\n';

  return 0;
}
