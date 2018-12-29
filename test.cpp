#include "Tensor.h"
#include <assert.h>

template <typename T>
void test_log(){
    Tape<T> tape;
    Tensor<T> x = Tensor<T>();
    x.root(&tape, 0.5);
    Tensor<T> z = x.log();
    z.grad();

    assert(z.wrt(x) == T(1.0)/x.value);
}

template <typename T>
void test_sin(){
    Tape<T> tape;
    Tensor<T> x = Tensor<T>();
    x.root(&tape, 0.5);
    Tensor<T> z = x.sin();
    z.grad();

    assert(z.wrt(x) == std::cos(x.value));
}

template <typename T>
void test_cos(){
    Tape<T> tape;
    Tensor<T> x = Tensor<T>();
    x.root(&tape, 0.5);
    Tensor<T> z = x.cos();
    z.grad();

    assert(z.wrt(x) == -std::sin(x.value));
}

template <typename T>
void test_pow(){
    Tape<T> tape;
    double ten = 10;
    Tensor<T> x = Tensor<T>();
    x.root(&tape, 0.5);
    Tensor<T> z = x.pow(ten);
    z.grad();

    assert(z.wrt(x) == ten * std::pow(x.value, ten - 1));
}

template <typename T>
void test_expr(){
    Tape<T> tape;
    Tensor<T> x = Tensor<T>();
    x.root(&tape, 0.5);
    Tensor<T> y = Tensor<T>();
    y.root(&tape, 4.2);
    Tensor<T> p = Tensor<T>();
    p.root(&tape, 4.2);
    Tensor<T> r = x * y.sin() + x.log();
    Tensor<T> z = r * r * r;
    z.grad();
    Tensor<T> q = z * z;
    q.grad();
    
    assert(q.value == z.value * z.value);
    assert(q.wrt(z) == T(2) * z.value);
    assert(z.value  == r.value * r.value * r.value);
    assert(z.wrt(r) == T(3) * std::pow(r.value, T(2)) );
    assert(z.wrt(p) == 0); // sanity check. 
    assert( std::abs(z.wrt(y) - T(3) * T(std::pow(r.value, 2.0)) * (x.value * std::cos(y.value)) ) < 1e-15 );
    assert( std::abs(z.wrt(x) - T(3) * T(std::pow(r.value, 2.0)) * ( std::sin(y.value) + (T(1.0)/x.value)) ) < 1e-15 );
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

