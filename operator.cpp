#pragma once

#include <tuple>
#include <typeindex>
#include <xtensor.hpp>
using namespace std;


// template<typename T>
// struct OperatorBase {
//     virtual ~OperatorBase() = default;
//     template <typename Derived>
//     virtual void backward(xt::xarray<T>& grad) = 0;
//     virtual void propagate() = 0;
// };


template <typename T, typename... Args>
class Operator {
private:
    std::tuple<Args&...> inputs;

public:
    Operator(Args&&... args) : inputs(std::forward<Args>(args)...) {}

    std::tuple<Args&...>& getInputs() { return inputs; }

    template <typename Derived>
    void backward(xt::xarray<T>& grad) {
        Derived::backward(*this, grad);
    }

    // call backward() on each input 
    void propagate() {
        std::apply([&](auto&... args){
            ((args.getSource() != nullptr ? args.backward(const_cast<xt::xarray<T>&>(args.getGrad())) : void()), ...);
        }, inputs);
    }


};