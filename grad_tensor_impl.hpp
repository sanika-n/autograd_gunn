#pragma once
#include "grad_tensor_decl.hpp"
#include <xtensor-blas/xlinalg.hpp>
#include "fw_bw.cpp"



template <typename T>
GradTensor<T>& GradTensor<T>::operator+ (GradTensor<T>& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    
    auto ret = (GradFunction<T>::template apply<Add<T>>(*this, other));
    return *ret;
}

template <typename T>
GradTensor<T>& GradTensor<T>::operator+ (GradTensor<T>&& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    auto ret = GradFunction<T>::template apply<Add<T>>(*this, std::move(other));
    return *ret;
}

template <typename T>
GradTensor<T>& GradTensor<T>::operator- (GradTensor<T>& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    //return Add<T>::template apply<T>(*this, other);
    auto ret = GradFunction<T>::template apply<Sub<T>>(*this, other);
    return *ret;
}

template <typename T>
GradTensor<T>& GradTensor<T>::operator- (GradTensor<T>&& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for +");
    }
    auto ret = GradFunction<T>::template apply<Sub<T>>(*this, std::move(other));
    return *ret;
}

// template <typename T>
// auto GradTensor<T>::operator* (GradTensor<T> other) {
//     if (data.shape() != other.data.shape()) {
//         throw std::invalid_argument("Shape mismatch for *");
//     }
//     //return Add<T>::template apply<T>(*this, other);
//     return GradFunction<T>::template apply<Mult<T>>(*this, other);
// }

template <typename T>
GradTensor<T>& GradTensor<T>::operator* (GradTensor<T>& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for *");
    }
    //return Add<T>::template apply<T>(*this, other);
    auto ret = GradFunction<T>::template apply<Mult<T>>(*this, other);
    return *ret;
}

template <typename T>
GradTensor<T>& GradTensor<T>::operator* (GradTensor<T>&& other) {
    if (data.shape() != other.data.shape()) {
        throw std::invalid_argument("Shape mismatch for *");
    }
    auto ret = GradFunction<T>::template apply<Mult<T>>(*this, std::move(other));
    return *ret;
}

template <typename T>
GradTensor<T>::~GradTensor() {
    delete source;
}