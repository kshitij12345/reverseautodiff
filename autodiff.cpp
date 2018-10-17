#include "Tensor.h"
#include <assert.h>

void test_log(){
    Tape tape;
    Tensor x = Tensor();
    x.root(&tape, 0.5);
    Tensor z = x.log();
    z.grad();

    assert(z.wrt(x) == 1.0/x.value);
}

void test_sin(){
    Tape tape;
    Tensor x = Tensor();
    x.root(&tape, 0.5);
    Tensor z = x.sin();
    z.grad();

    assert(z.wrt(x) == std::cos(x.value));
}

void test_cos(){
    Tape tape;
    Tensor x = Tensor();
    x.root(&tape, 0.5);
    Tensor z = x.cos();
    z.grad();

    assert(z.wrt(x) == -std::sin(x.value));
}

void test_expr(){
    Tape tape;
    Tensor x = Tensor();
    x.root(&tape, 0.5);
    Tensor y = Tensor();
    y.root(&tape, 4.2);
    Tensor p = Tensor();
    p.root(&tape, 4.2);
    Tensor z = x * y.sin() + x.log();
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
    test_expr();
    std::cout << "Success\n";
    return 0;
}

