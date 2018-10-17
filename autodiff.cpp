#include "Tensor.h"
#include <assert.h>

int main()
{
    Tape tape;
    Tensor x = Tensor();
    x.root(&tape, 0.5);
    Tensor y = Tensor();
    y.root(&tape, 4.2);
    Tensor p = Tensor();
    p.root(&tape, 4.2);
    Tensor z = x * y + x.sin();
    z.grad();

    assert(z.value  == x.value*y.value + std::sin(x.value));
    assert(z.wrt(x) == y.value + cos(x.value));
    assert(z.wrt(y) == x.value);
    assert(z.wrt(p) == 0);

    std::cout << "Success\n";
    return 0;
}

