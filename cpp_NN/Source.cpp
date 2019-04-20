#include <iostream>
#include <time.h>
#include <math.h>
#include <random>
#include <fstream>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <random>
#ifndef DEBUG
#include <matplotlibcpp.h>
#endif
#include "cpp_nn_lr.h"


#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl 

// liner regression

using namespace Eigen;

#ifndef DEBUG
namespace plt = matplotlibcpp;
#endif


void generate_saple_data(int sample_size, int feature_dim, int num_classes, MatrixXd *v)
{
	static std::mt19937_64 mt64(0);
	std::uniform_int_distribution<int> lavel(0, 1);
	std::vector<int> list;
	/**
	for (int i = 0; i < v->rows(); i++) {
		list[i] = lavel(mt64);
	}
	*/
	// standard normal disribution
	std::random_device seed;
	std::mt19937 engine(seed());

	double mu = 0.0, sig = 1.0;
	// Initialize generation engine
	std::normal_distribution<> dist(mu, sig);

	// generate
	for (int j = 0; j < v->cols(); j++) {
		for (int i = 0; i < v->rows(); i++) {
			(*v)(i,j) =(dist(engine) + 3);
		}	
	}

	auto alc = std::allocator<double>();
	std::vector<double> x1(v->col(0).data(), v->col(0).data()+v->col(0).rows());
	std::vector<double> x2(v->col(1).data(), v->col(1).data() + v->col(1).rows());
#ifndef DEBUG
	plt::scatter(x1, x2, 10.0);
	plt::show();
#endif
}


int main() {
	// Data load Process

	Liner_Reg nn(2, 2);
	
	std::cout << "bias : " << nn.b << std::endl;
	PRINT_MAT(nn.W1);

	srand((unsigned int)time(0));
	Vector2d x1 = Vector2d::Random(2).cwiseAbs();
	int t1 = 1;

	double y = nn.liner_predict(x1.transpose());
	double loss = nn.loss(x1, 1);

	VectorXd dx1 = VectorXd::Random(2).cwiseAbs();
	nn.liner_num_grad_weight(x1, t1, &dx1);
	PRINT_MAT(dx1);

	MatrixXd sample_data = MatrixXd::Zero(32, 2);
	generate_saple_data(0, 0, 0, &sample_data);
	PRINT_MAT(sample_data);

	/*
	int input_dim = 2;
	int iterate_num = 10000;
	int lr = 0.1; // learning rate

	Liner_Reg nn(input_dim, 2);

	for (int i = 0; i < iterate_num; i++) {
		Vector2d x1 = Vector2d::Zero(2);
		int t1;
		// allocate train data

		// train the model
		MatrixXd dW = nn.W1.setZero();
		nn.liner_num_grad_weight(x1, t1, &dW);
		nn.W1 = -dW;
		nn.b = nn.b - lr*nn.liner_num_grad_bias(x1, t1);
	}
	*/
}