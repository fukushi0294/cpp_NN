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
std::vector<double> avgLoss;
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
	double loss_batch = 0;
	for (int i = 0; i < features->rows(); i++) {
		MatrixXd feature = features->row(i);
		MatrixXd lavel = lavels->row(i).transpose();
		MatrixXd dW = MatrixXd::Zero(LR->W1.rows(),LR->W1.cols());
		LR->liner_num_grad_weight(feature, lavel, &dW);
		LR->W1 = LR->W1 - (dW*learning_rate);
		MatrixXd db = MatrixXd::Zero(LR->b.rows(), LR->b.cols());
		LR->liner_num_grad_bias(feature, lavel, &db);
		LR->b = LR->b - (db*learning_rate);
		loss_batch += LR->loss(feature, lavel);
	}
#ifndef DEBUG
	Loss.push_back(loss_batch/features->rows());
#endif // !DEBUG
}

#ifndef DEBUG
static double print_training_progress(int mb, int frequency)
{
	double training_loss = 0;
	if ((mb % frequency) == 0) {
		training_loss = Loss[mb];
	}
	return training_loss;
}

std::vector<double> moving_average(std::vector<double> training_loss, int w=20) {
	if (training_loss.size() < w) {
		return training_loss;
	}
	std::vector<double> val;
	for (int i=0; i < training_loss.size(); i++) {
		if (i < w) {
			val.push_back(training_loss[i]);
		}
		else {
			double tmp_sum = 0;
			for (int j = i-w ; j <= i; j++) {
				tmp_sum += training_loss[j];
			}
			val.push_back(tmp_sum / w);
		}
	}
	return val;
}
#endif // !DEBUG


int main() {
	// Input random parameter
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

	// Result before training
	double loss = nn.loss(x1,t1);
	std::cout << "Loss : " << loss << std::endl;

	int input_dim = 2;
	int batchsize = 25;
	int num_sample_to_train = 200000;
	int iterate_num = num_sample_to_train/batchsize;
	double lr = 0.1; // learning rate

	MatrixXd sample_features = MatrixXd::Zero(batchsize, input_dim);
	MatrixXd sample_lavels = MatrixXd::Zero(batchsize, input_dim);
	generate_sample_data(&sample_features, &sample_lavels);

#ifndef DEBUG // plot sample data
	std::vector<double> sample_x1(sample_features.col(0).data(), sample_features.col(0).data() + sample_features.col(0).rows());
	std::vector<double> sample_x2(sample_features.col(1).data(), sample_features.col(1).data() + sample_features.col(1).rows());
	plt::scatter(sample_x1, sample_x2, 5.0);
	plt::show();
#endif

	for (int i = 0; i<iterate_num; i++) {
		generate_sample_data(&sample_features, &sample_lavels);
		batch_train_sample(&nn, &sample_features, &sample_lavels, lr);
#ifndef DEBUG
		double ret = 0;
		if ((ret=print_training_progress(i, 50) )!= 0) {
			avgLoss.push_back(ret);
		}
#endif
	}

	PRINT_MAT(nn.W1);
	PRINT_MAT(nn.b);
	
	MatrixXd feature = sample_features.row(0);
	MatrixXd lavel = sample_lavels.row(0).transpose();
	std::cout << "Final loss : " << nn.loss(feature, lavel) << std::endl;

#ifndef DEBUG
	plt::plot(moving_average(avgLoss));
	plt::show();
#endif
	
	// Evalute model
	MatrixXd eval_features = MatrixXd::Zero(batchsize, input_dim);
	MatrixXd eval_lavels = MatrixXd::Zero(batchsize, input_dim);
	MatrixXd predict_lavels = MatrixXd::Zero(batchsize, input_dim);
	generate_sample_data(&eval_features, &eval_lavels);
	for (int i = 0; i < batchsize; i++) {
		nn.liner_predict(eval_features.row(i));
		predict_lavels.row(i) = nn.y.transpose();
	}

	std::cout << "Lavels : [";
	for (int i = 0; i < batchsize; i++) {
		std::cout << eval_lavels(i, 0) << ", ";
	}
	std::cout << "]" << std::endl;

	double accurerancy = 0;
	std::cout << "Predicts : [";
	for (int i = 0; i < batchsize; i++) {
		int lavel = 0;
		if (predict_lavels(i, 0) > 0.5) {
			lavel = 1;
		}
		else {
			lavel = 0;
		}
		std::cout << lavel << ", ";
		if (lavel==eval_lavels(i,0))
		{
			accurerancy++;
		}
	}
	std::cout << "]" << std::endl;
	std::cout << "Accurerancy : " << accurerancy / batchsize << std::endl;
}