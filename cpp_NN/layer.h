#pragma once
#include <math.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

class Relu
{
public:
	Relu();
	~Relu();
	Eigen::MatrixXd forward();
	Eigen::MatrixXd backward();
private:
	Eigen::MatrixXd mask;
};

Relu::Relu()
{
}

Relu::~Relu()
{
}

Eigen::MatrixXd Relu::forward() {

}

Eigen::MatrixXd Relu::backward() {

}