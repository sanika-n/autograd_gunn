#pragma once
#include <xtensor.hpp>
#include <tuple>
#include <iostream>
#include <type_traits>
using namespace std;

// template<typename T>
// class OperatorBase;

template<typename T>
class GradFunction;

template <typename T>
class GradTensor {
private:
    xt::xarray<T> data;
    xt::xarray<T> grad;
    GradFunction<T>* source;

public:
    GradTensor() : data(), grad(), source(nullptr) {}
    GradTensor(const xt::xarray<T>& d) : data(d), grad(xt::zeros_like(d)), source(nullptr) {}
    ~GradTensor();

    const xt::xarray<T>& getData() const { return data; }
    const xt::xarray<T>& getGrad() const { return grad; }
    void add_grad(const xt::xarray<T>& g) { grad += g; }

    void setSource(GradFunction<T>* op_ptr) { source = op_ptr; }
    GradFunction<T>* getSource() const { return source; }

    GradTensor<T>& operator+(GradTensor<T>& other);
    GradTensor<T>& operator+(GradTensor<T>&& other);
    GradTensor<T>& operator-(GradTensor<T>& other);
    GradTensor<T>& operator-(GradTensor<T>&& other);
    GradTensor<T>& operator*(GradTensor<T>& other);
    GradTensor<T>& operator*(GradTensor<T>&& other);
    GradTensor<T>& operator/(GradTensor<T>& other);
    GradTensor<T>& operator/(GradTensor<T>&& other);

    void backward(const xt::xarray<T>& grad_current) {
        if (source != nullptr) {
            source->backward(const_cast<xt::xarray<T>&>(grad_current));
            source->propagate();
        } else {
            cout << "leaf (no source)" << endl;
        }
    }
};