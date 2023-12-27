#pragma once

#include <Eigen/Core>
#include <vector>
#include "Optimizer.h"
#include "RNG.h"
#include "Config.h"

class Layer
{
protected:
	const int m_in_size;
	const int m_out_size;

	//typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::MatrixXd Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	virtual void init(const Scalar& mu, const Scalar& sigma, RNG& rng) = 0;
	virtual void forward(const Matrix& prev_layer_output) = 0;
	
	virtual const Matrix& output() const = 0;

	virtual void backprop() = 0;
	virtual const Matrix& backprop_data(const Matrix& pre_layer_output, const Matrix& next_layer_data) const = 0;

	virtual std::vector<Scalar> get_parameters() const = 0;
	virtual void set_parameters(const std::vector<Scalar>& param) {}
	virtual std::vector<Scalar> get_derivatives() const = 0;

public:
	Layer(const int in_size, const int out_size) :
		m_in_size(in_size), m_out_size(out_size) {}
	virtual ~Layer();

private:

};

Layer::Layer()
{
}

Layer::~Layer()
{
}