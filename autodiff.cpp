#include "Var.h"
#include <assert.h>

int main()
{
    Tape tape;
    Var x = Var();
    x.root_var(&tape, 0.5);
    Var y = Var();
    y.root_var(&tape, 4.2);
    Var p = Var();
    p.root_var(&tape, 4.2);
    Var z = x * y + x.sin();
    z.grad();

    assert(z.value  == x.value*y.value + std::sin(x.value));
    assert(z.wrt(x) == y.value + cos(x.value));
    assert(z.wrt(y) == x.value);
    assert(z.wrt(p) == 0);

    std::cout << "Success\n";
    return 0;
}

