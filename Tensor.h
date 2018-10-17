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

struct Tensor {

public:
    Tape* tape;
    int index;
    double value;
    std::vector<double> derivs;

    Tensor() {};
    Tensor(Tape* tape, int index, double value) : tape(tape), index(index), value(value) {
    };

    void Tensor::root(Tape* tape, double value) {
        this->tape = tape;
        this->index = tape->nodes.size();
        this->value = value;
        push0(tape);
        }

    double wrt(Tensor x){
        return this->derivs[x.index];
    }

    void Tensor::grad() {
        int length = this->tape->nodes.size();
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

    Tensor Tensor::sin() {
        int ind = push1(this->tape, this->index, std::cos(this->value));
        return Tensor(this->tape, ind, std::sin(this->value));
    }

    Tensor Tensor::cos() {
        int ind = push1(this->tape, this->index, -std::sin(this->value));
        return Tensor(this->tape, ind, std::cos(this->value));
    }

    Tensor Tensor::operator +(Tensor other) {
        int ind = push2(this->tape, this->index, 1.0, other.index, 1.0);
        return Tensor(this->tape, ind, this->value + other.value);
    }

    Tensor Tensor::operator -(Tensor other) {
        int ind = push2(this->tape, this->index, 1.0, other.index, 1.0);
        return Tensor(this->tape, ind, this->value - other.value);
    }

    Tensor Tensor::operator *(Tensor other) {
        int ind = push2(this->tape, this->index, other.value, other.index, this->value);
        return Tensor(this->tape, ind, this->value * other.value);
    }

    Tensor Tensor::operator /(Tensor other) {
        int ind = push2(this->tape, this->index, 1/other.value, other.index, this->value);
        return Tensor(this->tape, ind, this->value / other.value);
    }

};