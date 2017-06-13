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

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "tests/doctest.h"
#include "psimd/psimd.h"

using vfloat = psimd::pack<float>;
using vint   = psimd::pack<int>;
using vmask  = psimd::mask<DEFAULT_WIDTH>;

// pack<> operators ///////////////////////////////////////////////////////////

TEST_CASE("operator+()")
{
  vfloat v1(1.f), v2(2.f);

  REQUIRE(psimd::all((v1 + v2)  == vfloat(3.f)));
  REQUIRE(psimd::all((v1 + 2.f) == vfloat(3.f)));
  REQUIRE(psimd::all((2.f + v1) == vfloat(3.f)));

  // Add checks to make sure we don't promote regular math!
  bool value = std::is_same<decltype(1.f + 1.f), float>::value;
  value |= std::is_same<decltype(1.f + 1.), double>::value;
  value |= std::is_same<decltype(1.f + 1), float>::value;
  REQUIRE(value);
}

TEST_CASE("operator-()")
{
  vfloat v1(2.f), v2(1.f);

  REQUIRE(psimd::all((v1 - v2)  == vfloat(1.f)));
  REQUIRE(psimd::all((v1 - 2.f) == vfloat(0.f)));
  REQUIRE(psimd::all((4.f - v1) == vfloat(2.f)));

  // Add checks to make sure we don't promote regular math!
  bool value = std::is_same<decltype(1.f - 1.f), float>::value;
  value |= std::is_same<decltype(1.f - 1.), double>::value;
  value |= std::is_same<decltype(1.f - 1), float>::value;
  REQUIRE(value);
}

TEST_CASE("operator*()")
{
  vfloat v1(2.f), v2(1.f);

  REQUIRE(psimd::all((v1 * v2)  == vfloat(2.f)));
  REQUIRE(psimd::all((v1 * 2.f) == vfloat(4.f)));
  REQUIRE(psimd::all((2.f * v1) == vfloat(4.f)));

  // Add checks to make sure we don't promote regular math!
  bool value = std::is_same<decltype(1.f * 1.f), float>::value;
  value |= std::is_same<decltype(1.f * 1.), double>::value;
  value |= std::is_same<decltype(1.f * 1), float>::value;
  REQUIRE(value);
}

TEST_CASE("operator/()")
{
  vint v1(4), v2(2);

  REQUIRE(psimd::all((v1 / v2) == vint(2)));
  REQUIRE(psimd::all((v1 / 2)  == vint(2)));
  REQUIRE(psimd::all((8 / v1)  == vint(2)));

  // Add checks to make sure we don't promote regular math!
  bool value = std::is_same<decltype(1.f / 1.f), float>::value;
  value |= std::is_same<decltype(1.f / 1.), double>::value;
  value |= std::is_same<decltype(1.f / 1), float>::value;
  REQUIRE(value);
}

TEST_CASE("operator%()")
{
  vint v1(4), v2(3);

  REQUIRE(psimd::all((v1 % v2) == vint(1)));
  REQUIRE(psimd::all((v1 % 8)  == vint(4)));
  REQUIRE(psimd::all((8 % v1)  == vint(0)));

  // Add checks to make sure we don't promote regular math!
  bool value = std::is_same<decltype(1 % 1), int>::value;
  REQUIRE(value);
}

// pack<> algorithms //////////////////////////////////////////////////////////

TEST_CASE("foreach()")
{
  vfloat v1(0.f);
  vfloat v2(1.f);

  foreach(v1, [](float &l) { l = 1; });

  REQUIRE(psimd::all(v1 == v2));
}

TEST_CASE("any()")
{
  vmask m(0);
  REQUIRE(!psimd::any(m));
  m[0] = 1;
  REQUIRE(psimd::any(m));
}

TEST_CASE("all()")
{
  vmask m(0);
  REQUIRE(!psimd::all(m));
  m[0] = 1;
  REQUIRE(!psimd::all(m));
  foreach(m, [](int &l) { l = 1; });
  REQUIRE(psimd::all(m));
}
