
#include <iostream>
#include <mpl_all.hpp>
#include <set>
#include <indicators.hpp>


template <typename __IN_0, typename __IN_1, typename __IN_2, typename __IN_3, typename __IN_4, typename __IN_5>
struct __ {
  template <typename w5UE156, typename KkSRn9X, typename dYRk2kv>
  struct yW4Acq9 {
    template <typename G6tPwy1, typename QqjJSaD>
    struct laERA5K;

    template <typename QqjJSaD>
    struct HqhM6CQ : mpl::eval_if<typename mpl::apply<w5UE156, QqjJSaD>::type, laERA5K<KkSRn9X, QqjJSaD>, mpl::identity<QqjJSaD>> {};

    template <typename G6tPwy1, typename QqjJSaD>
    struct laERA5K {
      typedef typename HqhM6CQ<typename mpl::apply<G6tPwy1, QqjJSaD>::type>::type type;
    };

    typedef typename HqhM6CQ<dYRk2kv>::type type;
  };

  template <typename A, typename B, typename C>
  struct Q {
    template <typename a, typename b, typename c>
    struct _ {
      constexpr int operator()() { 
        int64_t i = a()();
        int64_t j = b()();
        int64_t k = c()();
        return ( (i % k) * (j % k) ) % k; }
    };
    typedef typename Q::_<A, B, C> type;
  };


  template <typename A, typename B, typename C>
  struct E {
    template <typename a, typename b, typename c>
    struct _ {
      constexpr int operator()() { 
        int64_t i = a()();
        int64_t j = b()();
        int64_t k = c()();
        return ( (i % k) + (j % k) ) % k; }
    };
    typedef typename E::_<A, B, C> type;
  };

  template <typename A, typename B>
  struct X {
    template <typename a, typename b>
    struct _ {
      constexpr auto operator()() { return a()() - b()(); }
    };
    typedef typename X::_<A, B> type;
  };

  template <typename A, typename B>
  struct C {
    template <typename a, typename b>
    struct _ {
      constexpr auto operator()() { 
        int i = a()();
        int n = b()();
        return (i % n + n) % n; 
      }
    };
    typedef typename C::_<A, B> type;
  };

  template <typename A>
  struct W {
    template <typename a>
    struct _ {
      constexpr auto operator()() { return a(); }
    };

    typedef typename W::_<A> type;
  };

  typedef mpl::map<>::type GFz6Y1t;
  typedef typename mpl::insert<GFz6Y1t, mpl::pair<mpl::integral_c<int, 0>, __IN_0>>::type v0;
  typedef typename mpl::insert<v0, mpl::pair<mpl::integral_c<int, 1>, __IN_1>>::type v1;
  typedef typename mpl::insert<v1, mpl::pair<mpl::integral_c<int, 2>, __IN_2>>::type v2;
  typedef typename mpl::insert<v2, mpl::pair<mpl::integral_c<int, 3>, __IN_3>>::type v3;
  typedef typename mpl::insert<v3, mpl::pair<mpl::integral_c<int, 4>, __IN_4>>::type v4;
  typedef typename mpl::insert<v4, mpl::pair<mpl::integral_c<int, 5>, __IN_5>>::type pZ9jZIC;

  typedef mpl::map<>::type ffu4G7F;
  typedef typename mpl::insert<ffu4G7F, mpl::pair<mpl::integral_c<int, 0>, mpl::integral_c<int, 1>>>::type h0;
  typedef typename mpl::insert<h0, mpl::pair<mpl::integral_c<int, 1>, mpl::integral_c<int, 26>>>::type h1;
  typedef typename mpl::insert<h1, mpl::pair<mpl::integral_c<int, 2>, mpl::integral_c<int, 676>>>::type h2;
  typedef typename mpl::insert<h2, mpl::pair<mpl::integral_c<int, 3>, mpl::integral_c<int, 17576>>>::type h3;
  typedef typename mpl::insert<h3, mpl::pair<mpl::integral_c<int, 4>, mpl::integral_c<int, 456976>>>::type h4;
  typedef typename mpl::insert<h4, mpl::pair<mpl::integral_c<int, 5>, mpl::integral_c<int, 11881376>>>::type EefRWi4;

  template <typename v1, typename h>
  struct ___ {
    template <typename gtJsThJ>
    struct v07In9Z {
      typedef typename mpl::at<gtJsThJ, mpl::integral_c<int, 0>>::type MJLsCfM;  // v1
      typedef typename mpl::at<gtJsThJ, mpl::integral_c<int, 1>>::type ZyuKpsl;  // h
      typedef typename mpl::at<gtJsThJ, mpl::integral_c<int, 2>>::type ret;     // ret
      typedef typename mpl::at<gtJsThJ, mpl::integral_c<int, 3>>::type EefRWi4;
      typedef typename mpl::at<ZyuKpsl, EefRWi4>::type IRz1THr;  // h[i]
      typedef typename mpl::at<MJLsCfM, EefRWi4>::type L08ALrC;  // v1[i]

      typedef typename __::W<mpl::integral_c<int, 488148229>>::type y;

      typedef typename __::W<mpl::integral_c<int, 97>>::type m;
      typedef typename __::W<mpl::integral_c<int, 26>>::type al;
      typedef typename __::X<L08ALrC, m>::type HZp2n5R;
      typedef typename __::C<HZp2n5R, al>::type OLOL;

      typedef typename __::W<IRz1THr>::type the;
      typedef typename __::Q<the, OLOL, y>::type FQPS6Yw;

      typedef typename __::E<ret, FQPS6Yw, y>::type fuNhFLO;
      typedef typename __::C<fuNhFLO, y>::type EEE;

      typedef typename mpl::plus<EefRWi4, mpl::integral_c<int, 1>>::type v2mpbUr;
      typedef typename mpl::vector<MJLsCfM, ZyuKpsl, EEE, v2mpbUr>::type type;
    };

    template <typename hToxYkv>
    struct B0dgYj2 {
      typedef typename mpl::at<hToxYkv, mpl::integral_c<int, 3>>::type Soc0KO6;
      typedef typename mpl::less<Soc0KO6, mpl::integral_c<int, 6>>::type IMRebhO;
      typedef typename IMRebhO::type type;
    };

    typedef typename __::W<mpl::integral_c<int, 0>>::type the;
    typedef mpl::vector<v1, h, the, mpl::integral_c<int, 0>> mM1Khzf;

    typedef typename yW4Acq9<B0dgYj2<mpl::_1>, mpl::quote1<v07In9Z>, mM1Khzf>::type x1hcMbm;

    typedef typename mpl::at<x1hcMbm, mpl::integral_c<int, 0>>::type lThEnUZ;
    typedef typename mpl::at<x1hcMbm, mpl::integral_c<int, 1>>::type d7RbIBN;
    typedef typename mpl::at<x1hcMbm, mpl::integral_c<int, 2>>::type g1qeoeP;
    typedef typename mpl::at<x1hcMbm, mpl::integral_c<int, 3>>::type twBldGD;
    typedef g1qeoeP type;
  };


  typedef typename ___<pZ9jZIC, EefRWi4>::type type;
};

char P[6];
struct H {
  int operator()() { return P[0]; }
};
struct J {
  int operator()() { return P[1]; }
};
struct K {
  int operator()() { return P[2]; }
};
struct L {
  int operator()() { return P[3]; }
};
struct M {
  int operator()() { return P[4]; }
};
struct N {
  int operator()() { return P[5]; }
};
struct OM {
  typedef __<H, J, K, L, M, N>::type type;
};

template <typename __IN_1>
struct _ {
  template <typename QqjJSaD, typename VfvVcIr, typename ipW9EwN>
  struct yW4Acq9 {
    template <typename PfaMmSv, typename kpoOCyT>
    struct NiTtyVA;

    template <typename kpoOCyT>
    struct sRwL85O : mpl::eval_if<typename mpl::apply<QqjJSaD, kpoOCyT>::type, NiTtyVA<VfvVcIr, kpoOCyT>, mpl::identity<kpoOCyT>> {};

    template <typename PfaMmSv, typename kpoOCyT>
    struct NiTtyVA {
      typedef typename sRwL85O<typename mpl::apply<PfaMmSv, kpoOCyT>::type>::type type;
    };

    typedef typename sRwL85O<ipW9EwN>::type type;
  };

  template <typename A, typename B, typename C>
  struct R {
    template <typename a, typename b, typename c>
    struct _ {
      constexpr int operator()() { 
        int64_t i = a()();
        int64_t j = b()();
        int64_t k = c()();
        return ( (i % k) * (j % k) ) % k; }
    };
    typedef typename R::_<A, B, C> type;
  };

  template <typename A, typename B>
  struct E {
    template <typename a, typename b>
    struct _ {
      constexpr int operator()() { return a()() * b()(); }
    };
    typedef typename E::_<A, B> type;
  };

  template <typename A, typename B>
  struct D {
    template <typename a, typename b>
    struct _ {
      constexpr int operator()() { return a()() % b()(); }
    };
    typedef typename D::_<A, B> type;
  };

  template <typename A>
  struct __ {
    template <typename a>
    struct _ {
      constexpr int operator()() { return a(); }
    };

    typedef typename __::_<A> type;
  };

  template <typename GFz6Y1t, typename EwL56nG, typename isiWgNZ>
  struct q6ITZM5 {
    template <typename jtgUe52>
    struct RvEJgwB {
      typedef typename mpl::at<jtgUe52, mpl::integral_c<int, 0>>::type uNO6n9J;
      typedef typename mpl::at<jtgUe52, mpl::integral_c<int, 1>>::type EC3HqdZ;
      typedef typename mpl::at<jtgUe52, mpl::integral_c<int, 2>>::type J6afU1z;
      typedef typename mpl::at<jtgUe52, mpl::integral_c<int, 3>>::type T0YQOaN;
      typedef mpl::vector<uNO6n9J, EC3HqdZ, J6afU1z, T0YQOaN> F03vpUu;

      template <typename T3em6Ko>
      struct pZ9jZIC {
        typedef typename mpl::at<T3em6Ko, mpl::integral_c<int, 3>>::type ffu4G7F;
        typedef typename mpl::modulus<ffu4G7F, mpl::integral_c<int, 2>>::type gtJsThJ;
        typedef typename mpl::equal_to<gtJsThJ, mpl::integral_c<int, 1>>::type v07In9Z;
        typedef typename v07In9Z::type type;
      };

      template <typename MJLsCfM>
      struct ZyuKpsl {
        typedef typename mpl::at<MJLsCfM, mpl::integral_c<int, 0>>::type m0lcNQq;
        typedef typename mpl::at<MJLsCfM, mpl::integral_c<int, 1>>::type EefRWi4;
        typedef typename mpl::at<MJLsCfM, mpl::integral_c<int, 2>>::type j7c1f5S;
        typedef typename _::__<j7c1f5S>::type the;
        typedef typename _::R<m0lcNQq, EefRWi4, the>::type IRz1THr;
        typedef typename _::D<IRz1THr, the>::type HZp2n5R;
        typedef typename mpl::vector<HZp2n5R, EefRWi4, j7c1f5S, T0YQOaN>::type type;
      };

      struct L08ALrC {
        typedef typename ZyuKpsl<F03vpUu>::type type;
      };

      typedef typename mpl::eval_if<typename pZ9jZIC<F03vpUu>::type, L08ALrC, F03vpUu>::type FQPS6Yw;

      typedef typename mpl::at<FQPS6Yw, mpl::integral_c<int, 0>>::type fuNhFLO;
      typedef typename mpl::at<FQPS6Yw, mpl::integral_c<int, 1>>::type v2mpbUr;
      typedef typename mpl::at<FQPS6Yw, mpl::integral_c<int, 2>>::type hToxYkv;
      typedef typename mpl::at<FQPS6Yw, mpl::integral_c<int, 3>>::type B0dgYj2;
      typedef typename mpl::divides<B0dgYj2, mpl::integral_c<int, 2>>::type Soc0KO6;
      typedef typename _::__<hToxYkv>::type the;
      typedef typename _::R<v2mpbUr, v2mpbUr, the>::type IMRebhO;
      typedef typename _::D<IMRebhO, the>::type mM1Khzf;
      typedef typename mpl::vector<fuNhFLO, mM1Khzf, hToxYkv, Soc0KO6>::type type;
    };

    template <typename x1hcMbm>
    struct lThEnUZ {
      typedef typename mpl::at<x1hcMbm, mpl::integral_c<int, 3>>::type d7RbIBN;
      typedef typename mpl::greater<d7RbIBN, mpl::integral_c<int, 0>>::type g1qeoeP;
      typedef typename g1qeoeP::type type;
    };

    typedef typename _::__<mpl::integral_c<int, 1>>::type the;
    typedef mpl::vector<the, GFz6Y1t, isiWgNZ, EwL56nG> twBldGD;

    typedef typename yW4Acq9<lThEnUZ<mpl::_1>, mpl::quote1<RvEJgwB>, twBldGD>::type cMgSzmq;

    typedef typename mpl::at<cMgSzmq, mpl::integral_c<int, 0>>::type w5UE156;
    typedef typename mpl::at<cMgSzmq, mpl::integral_c<int, 1>>::type KkSRn9X;
    typedef typename mpl::at<cMgSzmq, mpl::integral_c<int, 2>>::type dYRk2kv;
    typedef typename mpl::at<cMgSzmq, mpl::integral_c<int, 3>>::type HqhM6CQ;
    typedef w5UE156 type;
  };
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 398527879>, mpl::integral_c<int, 521206709>>::type _0;
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 293888053>, mpl::integral_c<int, 280983943>>::type _1;
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 470714879>, mpl::integral_c<int, 481821731>>::type _2;
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 379903019>, mpl::integral_c<int, 446513681>>::type _3;
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 412387981>, mpl::integral_c<int, 295950349>>::type _4;
typedef typename q6ITZM5<__IN_1, mpl::integral_c<int, 461497417>, mpl::integral_c<int, 519344851>>::type _5;
};


int O0;

struct S {
  int operator()() { return O0; }
};

int mod(int a, int n){
  return (a % n + n)%n;
}


inline int table(int i) {
  O0 = OM::type()();
  // std::cout << P << " O0:" << O0 << "\n";
  switch (i) {
    case 0:
      return _<S>::_0()();
    case 1:
      return _<S>::_1()();
    case 2:
      return _<S>::_2()();
    case 3:
      return _<S>::_3()();
    case 4:
      return _<S>::_4()();
    case 5:
      return _<S>::_5()();
  }
  return 0;
}



int main() {
  std::string w;


  int ans[] = {316196015, 183449189, 325026406, 93125040, 247649200, 358564396};
  std::set<char> char_set = 
  {'m', 'g', 'z', 'r', 'e', 'a', 'b', '_', 'f', 'w', '{', 'p', '}', 'd', 'q', 'l', 'n', 'x', 's', 't', 'v', 'j', 'y', 'o', 'h', 'i', 'u', 'c', 'k'};
  // string.ascii_lowercase + '_' + '{' + '}'

  using namespace indicators;
  BlockProgressBar bar{
    option::BarWidth{80},
    option::MaxProgress{35},
    option::Start{"["},
    option::End{"]"},
    option::PostfixText{"Calculating..."},
    option::ForegroundColor{Color::cyan},
    option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
  };

  std::cout << "Flag: ";
  std::cin >> w;

  bool ret = true;
  const int sp = 6;
  for (int i = 0; i < 36; i++) {
    P[i % sp] = w[i];
    if(char_set.find(w[i]) == char_set.end()){
      std::cout << "\nNo" << "\n";
      return 0;
    }
    bar.set_progress(i);
    if (i % sp == sp - 1) {
      int a = table(i / sp);
      ret &= (a == ans[i/sp]);
    }
  }

  std::cout << (ret ? "\nYes":"\nNo") << "\n";
}

