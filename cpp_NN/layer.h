#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

class Relu
{
public:
	template <class T> void forward(T& obj);
	void forward(double& obj);
	template <class T> void backward(T& obj);
};

template <class T> 
void Relu::forward(T& obj) {
	for (int i = 0; i < obj.rows(); i++) {
		for (int j = 0; j < obj.cols(); j++)
		{
			if (obj(i, j) <= 0) {
				obj(i, j) = 0;
			}
		}
	}
}

void Relu::forward(double& obj) {
	if (obj <= 0) {
		obj = 0;
	}
}

template <class T>
void Relu::backward(T& obj) {
	Relu::forward(obj);
}

class Sigmoid
{
public:
	template <class T> void forward(T& obj);
	void forward(double& obj);
	template <class T> void backward(T& obj);
};

template <class T> 
void Sigmoid::forward(T& obj) {
	for (int i = 0; i < obj.rows(); i++) {
		for (int j = 0; j < obj.cols(); j++) {
			obj(i, j) = 1 / (1 + std::exp(obj(i, j)));
		}
	}
}

void Sigmoid::forward(double& obj) {
	obj = 1 / (1 + std::exp(obj));
}

