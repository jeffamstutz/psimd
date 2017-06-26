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

#include <iostream>

#if 0
#include "psimd/psimd.h"

using vfloat = psimd::pack<float>;
using vint   = psimd::pack<int>;
using vmask  = psimd::mask<>;

int main()
{
  vfloat a(4.0f);
  vfloat b(5.6f);

  vfloat c = a * b + a * b - a * b / a * b;

  std::cout << "c[0] == " << c[0] << std::endl;

  return 0;
}
#elif 1

inline void announce()
{
  static int numCalls = 0;
  std::cout << "creating pack #" << numCalls++ << std::endl;
}

struct pack
{
  pack() { announce(); }
  pack(int v) : value(v) { announce(); }
  int value;
};

inline pack operator+(const pack &p1, const pack &p2)
{
  pack result;
  result.value = p1.value + p2.value;
  return result;
}

int main()
{
  pack v1(1), v2(4);
  pack v3 = v1 + v2 + v1 + v2;

  std::cout << "result == " << v3.value << std::endl;

  return 0;
}

#else
#include <boost/mpl/int.hpp>
#include <boost/proto/core.hpp>
#include <boost/proto/context.hpp>
namespace mpl = boost::mpl;
namespace proto = boost::proto;
using proto::_;

// This grammar describes which TArray expressions
// are allowed; namely, int and array terminals
// plus, minus, multiplies and divides of TArray expressions.
struct TArrayGrammar
  : proto::or_<
        proto::terminal< int >
      , proto::terminal< int[3] >
      , proto::plus< TArrayGrammar, TArrayGrammar >
      , proto::minus< TArrayGrammar, TArrayGrammar >
      , proto::multiplies< TArrayGrammar, TArrayGrammar >
      , proto::divides< TArrayGrammar, TArrayGrammar >
    >
{};

template<typename Expr>
struct TArrayExpr;

// Tell proto that in the TArrayDomain, all
// expressions should be wrapped in TArrayExpr<> and
// must conform to the TArrayGrammar
struct TArrayDomain
  : proto::domain<proto::generator<TArrayExpr>, TArrayGrammar>
{};

// Here is an evaluation context that indexes into a TArray
// expression, and combines the result.
struct TArraySubscriptCtx
  : proto::callable_context< TArraySubscriptCtx const >
{
    typedef int result_type;

    TArraySubscriptCtx(std::ptrdiff_t i)
      : i_(i)
    {}

    // Index array terminals with our subscript. Everything
    // else will be handled by the default evaluation context.
    int operator ()(proto::tag::terminal, int const (&data)[3]) const
    {
        return data[this->i_];
    }

    std::ptrdiff_t i_;
};

// Here is an evaluation context that prints a TArray expression.
struct TArrayPrintCtx
  : proto::callable_context< TArrayPrintCtx const >
{
    typedef std::ostream &result_type;

    TArrayPrintCtx() {}

    std::ostream &operator ()(proto::tag::terminal, int i) const
    {
        return std::cout << i;
    }

    std::ostream &operator ()(proto::tag::terminal, int const (&arr)[3]) const
    {
        return std::cout << '{' << arr[0] << ", " << arr[1] << ", " << arr[2] << '}';
    }

    template<typename L, typename R>
    std::ostream &operator ()(proto::tag::plus, L const &l, R const &r) const
    {
        return std::cout << '(' << l << " + " << r << ')';
    }

    template<typename L, typename R>
    std::ostream &operator ()(proto::tag::minus, L const &l, R const &r) const
    {
        return std::cout << '(' << l << " - " << r << ')';
    }

    template<typename L, typename R>
    std::ostream &operator ()(proto::tag::multiplies, L const &l, R const &r) const
    {
        return std::cout << l << " * " << r;
    }

    template<typename L, typename R>
    std::ostream &operator ()(proto::tag::divides, L const &l, R const &r) const
    {
        return std::cout << l << " / " << r;
    }
};

// Here is the domain-specific expression wrapper, which overrides
// operator [] to evaluate the expression using the TArraySubscriptCtx.
template<typename Expr>
struct TArrayExpr
  : proto::extends<Expr, TArrayExpr<Expr>, TArrayDomain>
{
    typedef proto::extends<Expr, TArrayExpr<Expr>, TArrayDomain> base_type;

    TArrayExpr( Expr const & expr = Expr() )
      : base_type( expr )
    {}

    // Use the TArraySubscriptCtx to implement subscripting
    // of a TArray expression tree.
    int operator []( std::ptrdiff_t i ) const
    {
        TArraySubscriptCtx const ctx(i);
        return proto::eval(*this, ctx);
    }

    // Use the TArrayPrintCtx to display a TArray expression tree.
    friend std::ostream &operator <<(std::ostream &sout, TArrayExpr<Expr> const &expr)
    {
        TArrayPrintCtx const ctx;
        return proto::eval(expr, ctx);
    }
};

// Here is our TArray terminal, implemented in terms of TArrayExpr
// It is basically just an array of 3 integers.
struct TArray
  : TArrayExpr< proto::terminal< int[3] >::type >
{
    explicit TArray( int i = 0, int j = 0, int k = 0 )
    {
        (*this)[0] = i;
        (*this)[1] = j;
        (*this)[2] = k;
        static int numConstructed = 0;
        std::cout << "numConstructed == " << numConstructed++ << std::endl;
    }

    // Here we override operator [] to give read/write access to
    // the elements of the array. (We could use the TArrayExpr
    // operator [] if we made the subscript context smarter about
    // returning non-const reference when appropriate.)
    int &operator [](std::ptrdiff_t i)
    {
        return proto::value(*this)[i];
    }

    int const &operator [](std::ptrdiff_t i) const
    {
        return proto::value(*this)[i];
    }

    // Here we define a operator = for TArray terminals that
    // takes a TArray expression.
    template< typename Expr >
    TArray &operator =(Expr const & expr)
    {
        // proto::as_expr<TArrayDomain>(expr) is the same as
        // expr unless expr is an integer, in which case it
        // is made into a TArrayExpr terminal first.
        return this->assign(proto::as_expr<TArrayDomain>(expr));
    }

    template< typename Expr >
    TArray &printAssign(Expr const & expr)
    {
        *this = expr;
        std::cout << *this << " = " << expr << std::endl;
        return *this;
    }

private:
    template< typename Expr >
    TArray &assign(Expr const & expr)
    {
        // expr[i] here uses TArraySubscriptCtx under the covers.
        (*this)[0] = expr[0];
        (*this)[1] = expr[1];
        (*this)[2] = expr[2];
        return *this;
    }
};

int main()
{
    TArray a(3,1,2);

    TArray b;

    std::cout << a << std::endl;
    std::cout << b << std::endl;

    b[0] = 7; b[1] = 33; b[2] = -99;

    TArray c(a);

    std::cout << c << std::endl;

    a = 0;

    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;

    a = b + c;

    std::cout << a << std::endl;

    a.printAssign(b+c*(b + 3*c));

    return 0;
}
#endif