#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace node {

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


	class Affine {
	public:
		Eigen::MatrixXd w;
		Eigen::MatrixXd b;
		Eigen::MatrixXd x;
		Eigen::MatrixXd dw;
		Eigen::MatrixXd db;

		Affine(int input_dim, int output_dim) {
			this->w = Eigen::MatrixXd::Zero(input_dim, output_dim);
			
		}
		Eigen::MatrixXd forward(Eigen::MatrixXd& x_in) {
			this->x = Eigen::MatrixXd::Zero(x_in.rows(), x_in.cols());
			this->b = Eigen::MatrixXd::Zero(w.cols(), x_in.cols());
			Eigen::MatrixXd out = x_in * w + b;
			return out;
		}
		Eigen::MatrixXd backward(Eigen::MatrixXd& out) {
			Eigen::MatrixXd dx = out * (this->w.transpose());
			this->dw = this->x.transpose() * out;
			this->db = out;
			return dx;
		}
	};

	class SoftmaxWithLoss {
	public:
		Eigen::MatrixXd loss;
		Eigen::MatrixXd y; // out
		Eigen::MatrixXd t; // train data

		Eigen::MatrixXd forward(Eigen::MatrixXd x_in, Eigen::MatrixXd train) {
			this->t = train;
			this->y = softmax(x_in);
			this->loss = cross_entropy_error(this->y, this->t);
		}

		Eigen::MatrixXd backward(Eigen::MatrixXd dout) {
			int batch_size = this->t.rows();
			Eigen::MatrixXd dx = (this->y - this->t) / batch_size;
			return dx;
		}
	};


}