#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

double sigmoid(double m)
{
	double y = 1 / (1 + std::exp(-m));
	return y;
}

void softmax(MatrixXd x, MatrixXd *y)
{
	double sum = 0;
	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {
			sum += std::exp(x(i, j));
		}
	}

	for (int i = 0; i < x.rows(); i++) {
		for (int j = 0; j < x.cols(); j++) {
			(*y)(i, j) = std::exp(x(i, j)) / sum;
		}
	}
}


double cross_entropy_err(MatrixXd x, MatrixXd t)
{
	double err = 0;
	double delta = 1e-4;
	MatrixXd log_x = (x.array() + delta).log();
	for (int i = 0; i < t.rows(); i++) {
		
		err = err+(-1*t(i, 0)*log_x(i, 0));
	}
	return err;
}

// Dynamically giving type , but this is not recommended... 
/*
template <class T>
MatrixXd liner_num_grad_weight_tmp(MatrixXd x, MatrixXd t, T obj, MatrixXd out) {
	int ret = 0;
	try
	{
		// Check the member of obj correspoding to out using pointer
		// Is it need to add one more member pointing all member ?
		for (int i = 0; sizeof(obj.p) / sizeof(obj.p[0]); i++) {
			if (obj.p[i] == out) {
				ret = i;
				break;
			}
		}
	}
	catch (const std::exception&)
	{
		std::cout << "something wrong method\n" << std::endl;
	}
 
	MatrixXd dout = MatrixXd::Zero(*(obj->p[ret]).rows(), *(obj->p[ret]).cols());
	double h = 1e-4;
	for (int i = 0; i < *(obj.p[ret]).rows(); i++) {
		for (int j = 0; j < *(obj.p[ret]).cols(); j++) {
			double tmp = *(obj.p[ret])(i, j);
			*(obj.p[ret])(i, j) = tmp + h;
			double tmp1 = obj.loss(x, t);
			*(obj.p[ret])(i, j) = tmp - h;
			double tmp2 = obj.loss(x, t);
			dW(i, j) = (tmp1 - tmp2) / (2 * h);
			*(obj.p[ret])(i, j) = tmp;
		}
	}
	return dout;
}
*/