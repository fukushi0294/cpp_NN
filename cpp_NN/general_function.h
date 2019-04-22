#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl 
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

double cross_entropy_err(double x, double t)
{
	double delta = 1e-7;
	double log_x = std::log(x + delta);
	return -1* t * log_x;
}

// cross entopy matrix
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