#pragma once

#include <xtensor.hpp>
#include <xtensor-blas/xlinalg.hpp>
#include "grad_tensor_decl.hpp"
#include "operator.cpp"

using namespace std;

template<typename T>
class GradFunction{
    public:
    GradFunction(GradTensor<T>& arg1, GradTensor<T>& arg2) : arg1(&arg1), arg2(&arg2) {}

    template <typename Derived>
    static GradTensor<T>* apply(GradTensor<T>& arg1, GradTensor<T>& arg2) {
        GradTensor<T>* temp = Derived::forward(arg1, arg2);
        auto* grad_ptr = new Derived(arg1, arg2);
        temp->setSource(grad_ptr);
        return temp;
    }

    virtual void backward(xt::xarray<T>& grad) = 0;

    void propagate() {
            if (arg1->getSource() != nullptr) {
                arg1->backward(const_cast<xt::xarray<T>&>(arg1->getGrad()));
            }
            if (arg2->getSource() != nullptr) {
                arg2->backward(const_cast<xt::xarray<T>&>(arg2->getGrad()));
            }
    }

    protected:
    GradTensor<T>* arg1;
    GradTensor<T>* arg2;
};

template<typename T>
class Add : public GradFunction<T> {

public:
    using Base = GradFunction<T>;

    Add(GradTensor<T>& arg1, GradTensor<T>& arg2) : Base(arg1, arg2) {}

    static GradTensor<T>* forward(GradTensor<T>& a, GradTensor<T>& b){
        xt::xarray<T> newData = a.getData() + b.getData();
        return new GradTensor<T>(newData);
    }

    void backward(xt::xarray<T>& grad) override {
        this->arg1->add_grad(grad);
        this->arg2->add_grad(grad);
    }
};


template<typename T>
class Sub : public GradFunction<T> {

public:
    using Base = GradFunction<T>;

    Sub(GradTensor<T>& arg1, GradTensor<T>& arg2) : Base(arg1, arg2) {}

    static GradTensor<T>* forward(GradTensor<T>& a, GradTensor<T>& b){
        xt::xarray<T> newData = a.getData() - b.getData();
        return new GradTensor<T>(newData);
    }

    void backward(xt::xarray<T>& grad) override {
        this->arg1->add_grad(grad);
        this->arg2->add_grad(-1 * grad);
    }
};

template<typename T>
class Mult: public GradFunction<T>{
    public:
    using Base = GradFunction<T>;
    
    Mult(GradTensor<T>& arg1, GradTensor<T>& arg2) : Base(arg1, arg2) {}

    static GradTensor<T>* forward(GradTensor<T>& a, GradTensor<T>& b){
        if (a.getData().shape() != b.getData().shape()) {
            throw std::invalid_argument("Shape mismatch for *");
        }
        xt::xarray<T> newData = a.getData() * b.getData();
        return new GradTensor<T>(newData);
    }

    
    void backward(xt::xarray<T>& grad) override {
        this->arg1->add_grad(this->arg2->getData() * grad);
        this->arg2->add_grad(this->arg1->getData() * grad);
    }
};