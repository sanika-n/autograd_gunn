#include "grad_tensor_decl.hpp"
#include "grad_tensor_impl.hpp"
#include<string>
#include <xtensor/xtensor_forward.hpp>

int main(){
    GradTensor<float> a(xt::xarray<float>{2.0});
    GradTensor<float> b(xt::xarray<float>{1.0});
    GradTensor<float> u(xt::xarray<float>{1.0});

    xt::xarray<float> start = {1.0};

    auto c = a + b;
    auto d = b + u;
    
    auto e = c * d;

    // calling the backward function on tensor e
    e.backward(start);

    cout << "The value of c now is: " << endl;
    cout << c.getData() << endl;

    cout << "The value of d now is: " << endl;
    cout << d.getData() << endl;

    cout << "The value of e now is: " << endl;
    cout << e.getData() << endl;

    cout << endl;

    cout << "The gradient at c is: " << endl;
    cout << c.getGrad() << endl;

    cout << "The gradient at d is: " << endl;
    cout << d.getGrad() << endl;

    cout << "The gradient at a is: " << endl;
    cout << a.getGrad() << endl;

    cout << "The gradient at b is: " << endl;
    cout << b.getGrad() << endl;
}



