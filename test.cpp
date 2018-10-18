#include "Tensor.h"
#include <assert.h>

void test_log(){
    Tape<double> tape;
    auto x = Tensor<double>();
    x.root(&tape, 0.5);
    auto z = x.log();
    z.grad();

    assert(z.wrt(x) == 1.0/x.value);
}

void test_sin(){
    Tape<double> tape;
    auto x = Tensor<double>();
    x.root(&tape, 0.5);
    auto z = x.sin();
    z.grad();

    assert(z.wrt(x) == std::cos(x.value));
}

void test_cos(){
    Tape<double> tape;
    auto x = Tensor<double>();
    x.root(&tape, 0.5);
    auto z = x.cos();
    z.grad();

    assert(z.wrt(x) == -std::sin(x.value));
}

void test_pow(){
    Tape<double> tape;
    double ten = 10;
    auto x = Tensor<double>();
    x.root(&tape, 0.5);
    auto z = x.pow(ten);
    z.grad();

    assert(z.wrt(x) == ten * std::pow(x.value, ten - 1));
}

void test_expr(){
    Tape<double> tape;
    auto x = Tensor<double>();
    x.root(&tape, 0.5);
    auto y = Tensor<double>();
    y.root(&tape, 4.2);
    auto p = Tensor<double>();
    p.root(&tape, 4.2);
    auto z = x * y.sin() + x.log();
    z.grad();

    assert(z.value  == x.value*std::sin(y.value) + std::log(x.value));
    assert(z.wrt(x) == std::sin(y.value) + 1.0/x.value);
    assert(z.wrt(y) == x.value*std::cos(y.value));
    assert(z.wrt(p) == 0); // sanity check.
}

int main()
{
    test_log();
    test_cos();
    test_sin();
    test_pow();
    test_expr();
    std::cout << "Success\n";
    return 0;
}

