#include <iostream>
#include <vector>
#include <math.h>


/********* Tape Structure *********************************/

// holds the information
// on which node/s does this
// node depend on. And their
// weight/s by which the gradient
// will be backpropped.
template <typename DataType>
struct Node {
    // id of node on which this node depends on
    std::vector<long> depends_on = std::vector<long>();

    // corresponding weights of the nodes on gradient
    std::vector<DataType> weights = std::vector<DataType>();
};

// Tape records the computations.
// Each node depends on previously
// computed node/s.
template <typename DataType>
struct Tape {
    std::vector<Node<DataType>> nodes;
};

/********* Helper Functions to record on Tape *********/

// For root nodes that depend on
// on no-one
template <typename DataType>
int push0(Tape<DataType>* tape) {
    int len = tape->nodes.size();
    Node<DataType> this_node;
    tape->nodes.push_back(this_node);
    return len;
}

// For unary operations
// eg. sin, cos, log, etc.
template <typename DataType>
int push1(Tape<DataType>* tape, int deps, DataType weight) {
    int len = tape->nodes.size();
    Node<DataType> this_node;
    
    this_node.depends_on.push_back(deps);
    this_node.weights.push_back(weight);
    
    tape->nodes.push_back(this_node);
    return len;
}

// For binary operations
// eg. +, -, *, etc.
template <typename DataType>
int push2(Tape<DataType>* tape,int deps0, DataType weight0, int deps1, DataType weight1) {
    int len = tape->nodes.size();
    Node<DataType> this_node;
    
    this_node.depends_on.push_back(deps0);
    this_node.weights.push_back(weight0);

    this_node.depends_on.push_back(deps1);
    this_node.weights.push_back(weight1);

    tape->nodes.push_back(this_node);
    return len;
}

/*******************************************************/

// Structure which holds actual data.
template <class DataType>
struct Tensor {

public:
    // Initialize tape as object.
    // Thus it will be automatically freed.
    // When it goes out of scope.
    Tape<DataType>* tape;
    int index;
    DataType value;
    std::vector<DataType> derivs;

    // Constructors
    Tensor<DataType>() {};
    Tensor<DataType>(Tape<DataType>* tape, int index, DataType value) : tape(tape), index(index), value(value) {
    };

    // make this tensor as root variable
    // function to make root variables
    // more verbose.
    void root(Tape<DataType>* tape, DataType value) {
        this->tape = tape;
        this->index = tape->nodes.size();
        this->value = value;
        push0(tape);
        }

    // gradient this tensor wrt x
    DataType wrt(Tensor<DataType> x){
        return this->derivs[x.index];
    }

    // Compute gradient from this Tensor
    void grad(DataType seed = 1.0) {
        int length = this->tape->nodes.size();
        std::vector<Node<DataType>> nodes = this->tape->nodes;
        std::vector<DataType>derivs(length);
        derivs[this->index] = seed;
        for (int i = derivs.size()-1; i > -1; i--) {
            Node<DataType> node = nodes[i];
            auto deriv = derivs[i];
            for (auto j = 0; j < node.depends_on.size(); j++){
                derivs[node.depends_on[j]] += node.weights[j] * deriv;
            }
        }
        
        this->derivs = derivs;
    };

    /* Operations that are recorded on Tape */
    Tensor<DataType> sin() {
        int ind = push1(this->tape, this->index, std::cos(this->value));
        return Tensor<DataType>(this->tape, ind, std::sin(this->value));
    }

    Tensor<DataType> cos() {
        int ind = push1(this->tape, this->index, -std::sin(this->value));
        return Tensor<DataType>(this->tape, ind, std::cos(this->value));
    }

    Tensor<DataType> log() {
        int ind = push1(this->tape, this->index, DataType(1.0)/this->value);
        return Tensor<DataType>(this->tape, ind, std::log(this->value));
    }

    Tensor<DataType> pow(double power) {
        int ind = push1(this->tape, this->index, DataType(power) * DataType( std::pow(this->value, power - 1)));
        return Tensor(this->tape, ind, std::pow(this->value, power));
    }

    Tensor<DataType> operator +(Tensor<DataType> other) {
        int ind = push2(this->tape, this->index, DataType(1.0), other.index, DataType(1.0));
        return Tensor<DataType>(this->tape, ind, this->value + other.value);
    }

    Tensor<DataType> operator -(Tensor<DataType> other) {
        int ind = push2(this->tape, this->index, DataType(1.0), other.index, DataType(1.0));
        return Tensor<DataType>(this->tape, ind, this->value - other.value);
    }

    Tensor<DataType> operator *(Tensor<DataType> other) {
        int ind = push2(this->tape, this->index, other.value, other.index, this->value);
        return Tensor<DataType>(this->tape, ind, this->value * other.value);
    }

    Tensor<DataType> operator /(Tensor<DataType> other) {
        int ind = push2(this->tape, this->index, DataType(1.0)/other.value, other.index, this->value);
        return Tensor<DataType>(this->tape, ind, this->value / other.value);
    }

};