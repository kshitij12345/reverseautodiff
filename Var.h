#include <iostream>
#include <vector>
#include <math.h>

typedef struct Node {
    std::vector<double> weights = std::vector<double> ();
    std::vector<double> depends_on = std::vector<double> ();
} Node;

struct Grad {
    std::vector<double> derivs;
};

struct Tape {
    std::vector<Node> nodes;
};

int len(Tape* tape) {
        return tape->nodes.size();
    }

int push0(Tape* tape) {
    int len = tape->nodes.size();
    Node node;
    node.depends_on.push_back(len);
    node.depends_on.push_back(len);
    node.weights.push_back(0);
    node.weights.push_back(0);
    tape->nodes.push_back(node);
    return len;
}

int push1(Tape* tape, int deps, double weight) {
    int len = tape->nodes.size();
    Node node;
    node.depends_on.push_back(deps);
    node.depends_on.push_back(len);
    node.weights.push_back(weight);
    node.weights.push_back(0);
    tape->nodes.push_back(node);
    return len;
}

int push2(Tape* tape,int deps0, double weight0, int deps1, double weight1) {
    int len = tape->nodes.size();
    Node node;
    node.depends_on.push_back(deps0);
    node.depends_on.push_back(deps1);
    node.weights.push_back(weight0);
    node.weights.push_back(weight1);
    tape->nodes.push_back(node);
    return len;
}

struct Var {

public:
    Tape* tape;
    int index;
    float value;
    std::vector<double> derivs;

    Var() {};
    Var(Tape* tape, int index, float value) : tape(tape), index(index), value(value) {
    };

    void Var::root_var(Tape* tape, double value) {
        this->tape = tape;
        this->index = tape->nodes.size();
        this->value = value;
        push0(tape);
        }

    double wrt(Var x){
        return this->derivs[x.index];
    }

    void Var::grad() {
        int length = len(this->tape);
        std::vector<Node> nodes = this->tape->nodes;
        std::vector<double>derivs(length);
        derivs[this->index] = 1.0;
        for (int i = derivs.size()-1; i > -1; i--) {
            Node node = nodes[i];
            double deriv = derivs[i];
            for (auto j = 0; j < node.depends_on.size(); j++){
                derivs[node.depends_on[j]] += node.weights[j] * deriv;
            }
        }
        
        this->derivs = derivs;
    };

    /* Operations */

    Var Var::sin() {
        int ind = push1(this->tape, this->index, std::cos(this->value));
        return Var(this->tape, ind, std::sin(this->value));
    }

    Var Var::cos() {
        int ind = push1(this->tape, this->index, -std::sin(this->value));
        return Var(this->tape, ind, std::cos(this->value));
    }

    Var Var::operator +(Var other) {
        int ind = push2(this->tape, this->index, 1.0, other.index, 1.0);
        return Var(this->tape, ind, this->value + other.value);
    }

    Var Var::operator -(Var other) {
        int ind = push2(this->tape, this->index, 1.0, other.index, 1.0);
        return Var(this->tape, ind, this->value - other.value);
    }

    Var Var::operator *(Var other) {
        int ind = push2(this->tape, this->index, other.value, other.index, this->value);
        return Var(this->tape, ind, this->value * other.value);
    }

    Var Var::operator /(Var other) {
        int ind = push2(this->tape, this->index, 1/other.value, other.index, this->value);
        return Var(this->tape, ind, this->value / other.value);
    }

};