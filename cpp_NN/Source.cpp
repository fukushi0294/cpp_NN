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
std::vector<double> Loss;
#endif


void generate_sample_data(MatrixXd *features, MatrixXd *lavels)
{
	static std::mt19937 mt64(0);
	std::uniform_int_distribution<int> lavel(0, 1);
	int a = lavel(mt64);
	
	for (int i = 0; i < lavels->rows(); i++) {
		(*lavels)(i,0) = lavel(mt64);
		if ((*lavels)(i, 0) == 0) {
			(*lavels)(i, 1) = 1;
		}
		else {
			(*lavels)(i, 1) = 0;
		}
	}
	// standard normal disribution
	std::random_device seed;
	std::mt19937 engine(seed());

	double mu = 0.0, sig = 1.0;
	// Initialize generation engine
	std::normal_distribution<> dist(mu, sig);

	// generate
	for (int j = 0; j < features->cols(); j++) {
		for (int i = 0; i < features->rows(); i++) {
			(*features)(i,j) =(dist(engine) + 3)*((*lavels)(i, 0) +1);
		}	
	}
}


void batch_train_sample(Liner_Reg *LR, MatrixXd *features, MatrixXd *lavels, double learning_rate) {
	assert(features->rows() == lavels->rows());
	for (int i = 0; i < features->rows(); i++) {
		MatrixXd feature = features->row(i);
		MatrixXd lavel = lavels->row(i).transpose();
		MatrixXd dW = MatrixXd::Zero(LR->W1.rows(),LR->W1.cols());
		LR->liner_num_grad_weight(feature, lavel, &dW);
		LR->W1 = LR->W1 - (dW*learning_rate);
		MatrixXd db = MatrixXd::Zero(LR->b.rows(), LR->b.cols());
		LR->liner_num_grad_bias(feature, lavel, &db);
		LR->b = LR->b - (db*learning_rate);
#ifndef DEBUG
		Loss.push_back(LR->loss(feature, lavel));
#endif // !DEBUG
	}
}

int main() {
	// Data load Process
	Liner_Reg nn(2, 2);
	
	PRINT_MAT(nn.W1);
	PRINT_MAT(nn.b);
	PRINT_MAT(nn.y);

	srand((unsigned int)time(0));
	MatrixXd x1 = MatrixXd::Random(1,2).cwiseAbs();
	MatrixXd t1 = MatrixXd::Zero(2,1);
	t1 << 0, 1;
	
	nn.liner_predict(x1);
	PRINT_MAT(nn.y);

	double loss = nn.loss(x1,t1);
	std::cout << "Loss : " << loss << std::endl;

	MatrixXd dx1 = MatrixXd::Zero(2, 2);
	nn.liner_num_grad_weight(x1, t1, &dx1);
	PRINT_MAT(dx1);
	
	MatrixXd sample_features = MatrixXd::Zero(32, 2);
	MatrixXd sample_lavels = MatrixXd::Zero(32, 2);
	generate_sample_data(&sample_features, &sample_lavels);

#ifndef DEBUG // plot sample data
	std::vector<double> sample_x1(sample_features.col(0).data(), sample_features.col(0).data() + sample_features.col(0).rows());
	std::vector<double> sample_x2(sample_features.col(1).data(), sample_features.col(1).data() + sample_features.col(1).rows());
	plt::scatter(sample_x1, sample_x2, 5.0);
	plt::show();
#endif

	int input_dim = 2;
	int iterate_num = 10000;
	double lr = 0.1; // learning rate

	for (int i = 0; i<100; i++) {
		generate_sample_data(&sample_features, &sample_lavels);
		batch_train_sample(&nn, &sample_features, &sample_lavels, lr);
	}

	PRINT_MAT(nn.W1);
	PRINT_MAT(nn.b);
	
#ifndef DEBUG
	plt::plot(Loss);
	plt::show();
#endif

}