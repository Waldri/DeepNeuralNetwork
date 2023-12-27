#pragma once

#include <Eigen/Core>
#include <vector>
#include <stdexcept>
#include "../Config.h"
#include "../Layer.h"

template<typename Activaton>
class FullyConnected :public Layer
{
private:
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
	typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

	Matrix m_weight;
	Vector m_bias;
	Matrix m_dw;
	Vector m_db;
	Matrix m_z;
	Matrix m_a;
	Matrix m_din;

public:
	FullyConnected(const int in_size, const int out_size):
		Layer(in_size, out_size)
	{}

	void init(const Scalar& mu, const Scalar& sigma, RNG& rng);

};
