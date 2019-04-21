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


void generate_saple_data(MatrixXd *features, MatrixXd *lavels)
{
	static std::mt19937 mt64(0);
	std::uniform_int_distribution<int> lavel(0, 1);
	int a = lavel(mt64);
	
	for (int i = 0; i < lavels->rows(); i++) {
		(*lavels)(i,0) = lavel(mt64);
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

int batch_train_sample(Liner_Reg *LR, MatrixXd *features, VectorXd *lavels, double learning_rate) {
	assert(features->rows() == lavels->size());
	for (int i = 0; i < features->rows(); i++) {
		VectorXd feature = features->row(i);
		int lavel = (*lavels)[i];
		VectorXd dW = VectorXd::Zero(LR->W1.rows());
		LR->liner_num_grad_weight(feature, lavel, &dW);
		LR->W1 = LR->W1 - (dW*learning_rate);
		LR->b = LR->b - learning_rate * LR->liner_num_grad_bias(feature, lavel);
	}
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

	VectorXd dx1 = VectorXd::Zero(2).cwiseAbs();
	nn.liner_num_grad_weight(x1, t1, &dx1);
	PRINT_MAT(dx1);

	MatrixXd sample_features = MatrixXd::Zero(32, 2);
	MatrixXd sample_lavels = MatrixXd::Zero(32, 1);
	generate_saple_data(&sample_features, &sample_lavels);

#ifndef DEBUG // plot sample data
	std::vector<double> sample_x1(sample_features.col(0).data(), sample_features.col(0).data() + sample_features.col(0).rows());
	std::vector<double> sample_x2(sample_features.col(1).data(), sample_features.col(1).data() + sample_features.col(1).rows());
	plt::scatter(sample_x1, sample_x2, 5.0);
	plt::show();
#endif


	int input_dim = 2;
	int iterate_num = 10000;
	double lr = 0.1; // learning rate
	VectorXd _sample_lavels = sample_lavels.col(0);

	// pick up sample data
	std::cout << "before" << std::endl;
	PRINT_MAT(nn.W1);
	std::cout << nn.b << std::endl;
	batch_train_sample(&nn, &sample_features, &_sample_lavels, lr);
	std::cout << "after" << std::endl;
	PRINT_MAT(nn.W1);
	std::cout << nn.b << std::endl;


	// allocate train data

	// train the model


}