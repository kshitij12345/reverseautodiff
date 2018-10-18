#include "Tensor.h"
#include <assert.h>

template <typename T>
void test_log(){
    Tape<T> tape;
    auto x = Tensor<T>();
    x.root(&tape, 0.5);
    auto z = x.log();
    z.grad();

    assert(z.wrt(x) == T(1.0)/x.value);
}

template <typename T>
void test_sin(){
    Tape<T> tape;
    auto x = Tensor<T>();
    x.root(&tape, 0.5);
    auto z = x.sin();
    z.grad();

    assert(z.wrt(x) == std::cos(x.value));
}

template <typename T>
void test_cos(){
    Tape<T> tape;
    auto x = Tensor<T>();
    x.root(&tape, 0.5);
    auto z = x.cos();
    z.grad();

    assert(z.wrt(x) == -std::sin(x.value));
}

template <typename T>
void test_pow(){
    Tape<T> tape;
    double ten = 10;
    auto x = Tensor<T>();
    x.root(&tape, 0.5);
    auto z = x.pow(ten);
    z.grad();

    assert(z.wrt(x) == ten * std::pow(x.value, ten - 1));
}

template <typename T>
void test_expr(){
    Tape<T> tape;
    auto x = Tensor<T>();
    x.root(&tape, 0.5);
    auto y = Tensor<T>();
    y.root(&tape, 4.2);
    auto p = Tensor<T>();
    p.root(&tape, 4.2);
    auto z = x * y.sin() + x.log();
    z.grad();

    assert(z.value  == x.value*std::sin(y.value) + std::log(x.value));
    assert(z.wrt(x) == std::sin(y.value) + T(1.0)/x.value);
    assert(z.wrt(y) == x.value*std::cos(y.value));
    assert(z.wrt(p) == 0); // sanity check.
}

template <typename T>
struct Test{
    void run_test(){
        test_log<T>();
        test_cos<T>();
        test_sin<T>();
        test_pow<T>();
        test_expr<T>();
    }
};

int main()
{
    Test<double>().run_test();

    Test<float>().run_test();

    std::cout << "Success\n";
    return 0;
}

