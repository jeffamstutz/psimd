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

namespace psimd {

  // FUTURE CONFIG ITEMS //////////////////////////////////////////////////////

#define DEFAULT_WIDTH 8

#ifdef _WIN32
#  define PSIMD_ALIGN(...) __declspec(align(__VA_ARGS__))
#else
#  define PSIMD_ALIGN(...) __attribute__((aligned(__VA_ARGS__)))
#endif

  // pack<> type //////////////////////////////////////////////////////////////

  template <typename T, int W = DEFAULT_WIDTH>
  struct PSIMD_ALIGN(16) pack
  {
    pack() = default;
    pack(T value);

    const T& operator[](int i) const;
          T& operator[](int i);

    // Data //

    T data[W];
    enum {SIZE = W};
  };

  template <int W = DEFAULT_WIDTH>
  using mask = pack<int, W>;

  // pack<> inlined members ///////////////////////////////////////////////////

  template <typename T, int W>
  inline pack<T, W>::pack(T value)
  {
    for(int i = 0; i < W; ++i)
      data[i] = value;
  }

  template <typename T, int W>
  inline const T& pack<T, W>::operator[](int i) const
  {
    return data[i];
  }

  template <typename T, int W>
  inline T& pack<T, W>::operator[](int i)
  {
    return data[i];
  }

  // pack<> operators /////////////////////////////////////////////////////////

  template <typename T, int W>
  inline pack<T, W> operator+(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    pack<T, W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = p1[i] + p2[i];

    return result;
  }

  template <typename T, int W>
  inline mask<W> operator==(const pack<T, W> &p1, const pack<T, W> &p2)
  {
    mask<W> result;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      result[i] = p1[i] == p2[i];

    return result;
  }

  // pack<> algorithms ////////////////////////////////////////////////////////

  template <typename T, int W, typename TASK_T>
  inline void foreach(pack<T, W> &p, TASK_T &&task)
  {
    #pragma omp simd
    for (int i = 0; i < W; ++i)
      task(p[i]);
  }

  template <int W>
  inline bool any(const mask<W> &m)
  {
    bool result = false;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (m[i])
        result = true;

    return result;
  }

  template <int W>
  inline bool all(const mask<W> &m)
  {
    bool result = true;

    #pragma omp simd
    for (int i = 0; i < W; ++i)
      if (!m[i])
        result = false;

    return result;
  }

} // ::psimd
