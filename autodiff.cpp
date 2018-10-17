#include "Var.h"

int main()
{
    Tape tape;
    Var x = Var();
    x.root_var(&tape, tape.nodes.size(), 0.5);
    Var y = Var();
    y.root_var(&tape, tape.nodes.size(), 4.2);
    Var p = Var();
    p.root_var(&tape, tape.nodes.size(), 4.2);
    Var z = x * y + x.sin();
    z.grad();

    std::cout << z.value << " should be " << "2.5794" << "\n";
    std::cout << z.wrt(x) << " should be " << y.value + cos(x.value) << "\n";
    std::cout << z.wrt(y) << " should be " << x.value << "\n";
    std::cout << z.wrt(p) << " should be 0" << "\n";
    
    return 0;
}

