#pragma once
#include <iostream>
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
	MatrixXd W1;
	MatrixXd b;
	MatrixXd y;

	Liner_Reg(int input_size, int output_size) {
		srand((unsigned int) time(0));
		this->W1 = MatrixXd::Random(input_size,output_size).cwiseAbs();
		this->b = MatrixXd::Random(output_size,1).cwiseAbs();
		this->y = MatrixXd::Zero(output_size,1);
	}

	~Liner_Reg();

	void liner_predict(MatrixXd x) {
		this->y = MatrixXd::Zero(2, 1);
		MatrixXd y1 = this->W1*(x.transpose()) + this->b;
		softmax(y1, &(this->y));
	}

	double loss(MatrixXd x, MatrixXd t) {
		this->liner_predict(x);
		return cross_entropy_err(this->y, t);
	}

	MatrixXd liner_num_grad_weight(MatrixXd x, MatrixXd t) {
		MatrixXd dW = MatrixXd::Zero(this->W1.rows(), this->W1.cols());
		double h = 1e-4;
		for (int i = 0; i < this->W1.rows(); i++) {
			for (int j = 0; j < this->W1.cols(); j++) {
				double tmp = this->W1(i, j); //escape
				this->W1(i, j) = tmp + h;
				double tmp1 = this->loss(x, t);
				this->W1(i, j) = tmp - h;
				double tmp2 = this->loss(x, t);
				dW(i, j) = (tmp1 - tmp2) / (2 * h);
				this->W1(i, j) = tmp;
			}
		}
		return dW;
	}

	MatrixXd liner_num_grad_bias(MatrixXd x, MatrixXd t) {
		MatrixXd db = MatrixXd::Zero(this->b.rows(), this->b.cols());
		double h = 1e-4;
		for (int i = 0; i < this->b.rows(); i++) {
			for (int j = 0; j < this->b.cols(); j++) {
				double tmp = this->b(i, j); //escape
				this->b(i, j) = tmp + h;
				double tmp1 = this->loss(x, t);
				this->b(i, j) = tmp - h;
				double tmp2 = this->loss(x, t);
				db(i, j) = (tmp1 - tmp2) / (2 * h);
				this->b(i, j) = tmp;
			}
		}
		return db;
	}

private:

};

Liner_Reg::~Liner_Reg()
{

}