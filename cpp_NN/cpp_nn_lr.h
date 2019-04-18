#pragma once

#include <math.h>
#include <random>
#include <time.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "general_function.h"


using namespace Eigen;

class Liner_Reg
{
public:
	VectorXd W1;
	double b;
	Liner_Reg(int input_size, int output_size) {
		srand((unsigned int) time(0));
		this->W1 = VectorXd::Random(input_size).cwiseAbs();
		static std::mt19937_64 mt64(0);
		std::uniform_real_distribution<double> random_bias(0, 1);
		this->b = std::abs(random_bias(mt64));
	}
	~Liner_Reg();
	double liner_predict(VectorXd x) {
		double y = this->W1.dot(x) + this->b;
		double y1 = sigmoid(y);
		return y1;
	}

	double loss(VectorXd x, int t) {
		double y = this->liner_predict(x);
		return cross_entropy_err(y, t);
	}


	void liner_num_grad_weight(VectorXd x, int t, VectorXd* dx) {
		double h = 1e-4;
		int rows = this->W1.rows();
		for (int i = 0; i < rows; i++) {
			double tmp = this->W1(i);
			this->W1(i) = tmp + h;
			double tmp1 = this->loss(x, t);
			this->W1(i) = tmp - h;
			double tmp2 = this->loss(x, t);
			(*dx)(i) = (tmp1 - tmp2) / (2 * h);
			this->W1(i) = tmp;
		}
	}

	double liner_num_grad_bias(MatrixXd x, int t) {
		double h = 1e-4;
		double tmp = this->b;
		this->b = tmp + h;
		double tmp1 = this->loss(x, t);
		this->b = tmp - h;
		double tmp2 = this->loss(x, t);
		this->b = tmp;
		return (tmp1 - tmp2) / (2 * h);
	}

private:

};

Liner_Reg::~Liner_Reg()
{
	this->W1.setZero();
}