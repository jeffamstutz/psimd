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
#  define RTCORE_ALIGN(...) __declspec(align(__VA_ARGS__))
#else
#  define RTCORE_ALIGN(...) __attribute__((aligned(__VA_ARGS__)))
#endif

  // pack<> type //////////////////////////////////////////////////////////////

  template <typename PACK_T, int WIDTH = DEFAULT_WIDTH>
  struct pack
  {
    PACK_T data[WIDTH];
    enum {SIZE = WIDTH};

    PACK_T& operator[](int i);
  };

  template<typename PACK_T, int WIDTH>
  inline PACK_T &pack<PACK_T, WIDTH>::operator[](int i)
  {
    return data[i];
  }

} // ::psimd
