#include <iostream>
#include <vector>
#include <math.h>

typedef std::vector<double> Grad;

// Node holds the information
// on which node/s does this
// node depend on. Add their
// weight/s by which the gradient
// will be backpropped.
template <typename T>
struct Node {
    std::vector<T> weights = std::vector<T> ();
    std::vector<long> depends_on = std::vector<long> ();
};

// Tape records the computations.
// Each node depends on previously
// computed node/s.
template <typename T>
struct Tape {
    std::vector<Node<T>> nodes;
};

/********* Helper Functions to record on Tape *********/
template <typename T>
int push0(Tape<T>* tape) {
    int len = tape->nodes.size();
    Node<T> node;
    tape->nodes.push_back(node);
    return len;
}

template <typename T>
int push1(Tape<T>* tape, int deps, T weight) {
    int len = tape->nodes.size();
    Node<T> node;
    node.depends_on.push_back(deps);
    node.weights.push_back(weight);
    tape->nodes.push_back(node);
    return len;
}

template <typename T>
int push2(Tape<T>* tape,int deps0, T weight0, int deps1, T weight1) {
    int len = tape->nodes.size();
    Node<T> node;
    node.depends_on.push_back(deps0);
    node.depends_on.push_back(deps1);
    node.weights.push_back(weight0);
    node.weights.push_back(weight1);
    tape->nodes.push_back(node);
    return len;
}

/*******************************************************/

// Structure which holds actual data.
template <class T>
struct Tensor {

public:
    Tape<T>* tape;
    int index;
    T value;
    std::vector<T> derivs;

    // Constructors
    Tensor<T>() {};
    Tensor<T>(Tape<T>* tape, int index, T value) : tape(tape), index(index), value(value) {
    };

    // make this tensor as root variable
    void root(Tape<T>* tape, T value) {
        this->tape = tape;
        this->index = tape->nodes.size();
        this->value = value;
        push0(tape);
        }

    // gradient this tensor wrt x
    double wrt(Tensor<T> x){
        return this->derivs[x.index];
    }

    // Compute gradient from this Tensor
    void grad(T seed = 1.0) {
        int length = this->tape->nodes.size();
        std::vector<Node<T>> nodes = this->tape->nodes;
        std::vector<T>derivs(length);
        derivs[this->index] = seed;
        for (int i = derivs.size()-1; i > -1; i--) {
            Node<T> node = nodes[i];
            auto deriv = derivs[i];
            for (auto j = 0; j < node.depends_on.size(); j++){
                derivs[node.depends_on[j]] += node.weights[j] * deriv;
            }
        }
        
        this->derivs = derivs;
    };

    /* Operations that are recorded on Tape */
    Tensor<T> sin() {
        int ind = push1(this->tape, this->index, std::cos(this->value));
        return Tensor<T>(this->tape, ind, std::sin(this->value));
    }

    Tensor<T> cos() {
        int ind = push1(this->tape, this->index, -std::sin(this->value));
        return Tensor<T>(this->tape, ind, std::cos(this->value));
    }

    Tensor<T> log() {
        int ind = push1(this->tape, this->index, T(1.0)/this->value);
        return Tensor<T>(this->tape, ind, std::log(this->value));
    }

    Tensor<T> pow(double power) {
        int ind = push1(this->tape, this->index, T(power) * T(std::pow(this->value, power - 1)));
        return Tensor(this->tape, ind, std::pow(this->value, power));
    }

    Tensor<T> operator +(Tensor<T> other) {
        int ind = push2(this->tape, this->index, T(1.0), other.index, T(1.0));
        return Tensor<T>(this->tape, ind, this->value + other.value);
    }

    Tensor<T> operator -(Tensor<T> other) {
        int ind = push2(this->tape, this->index, T(1.0), other.index, T(1.0));
        return Tensor<T>(this->tape, ind, this->value - other.value);
    }

    Tensor<T> operator *(Tensor<T> other) {
        int ind = push2(this->tape, this->index, other.value, other.index, this->value);
        return Tensor<T>(this->tape, ind, this->value * other.value);
    }

    Tensor<T> operator /(Tensor<T> other) {
        int ind = push2(this->tape, this->index, T(1.0)/other.value, other.index, this->value);
        return Tensor<T>(this->tape, ind, this->value / other.value);
    }

};