#pragma once

#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>


double sigmoid(double m)
{
	double y = 1 / (1 + std::exp(-m));
	return y;
}

/*
void sigmoid_m(MatrixXd m, MatrixXd *y)
{
	y = 1 / (1 + (-m).array().exp());
}
*/


double cross_entropy_err(double x, double t)
{
	double delta = 1e-7;
	double log_x = std::log(x + delta);
	return -1* t * log_x;
}

// cross entopy should be scalar.
/*
MatrixXd cross_entropy_err_m(MatrixXd x, MatrixXd t)
{
	double delta = 1e-7;
	MatrixXd log_x = (x.array() + delta).log();
	return (t * log_x);
}
*/