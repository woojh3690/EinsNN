#ifndef EINSNN_FULLY_CONNECTED_H_
#define EINSNN_FULLY_CONNECTED_H_

#include "Layer.h"
#include "../Config.h"
#include "../activation/Activation.h"
#include "../activation/NoneActivation.h"

namespace EinsNN
{
	class Fully_connected : public Layer
	{
	private:
		TensorD m_W, m_dw;
		TensorD m_b, m_db;
		TensorD m_z;
		TensorD m_a;

		Activation* m_activeFunc = nullptr;

	public:
		Fully_connected(const int input_size, const int output_size) :
			Layer(input_size, output_size)
		{
			m_activeFunc = new NoneActivation();
		}

		Fully_connected(const int input_size, const int output_size, Activation* activeFunc) :
			Layer(input_size, output_size), m_activeFunc(activeFunc)
		{}
		~Fully_connected() {}

		void init() override
		{
			TensorD W({ this->m_in_size, this->m_out_size }, 1);
			TensorD b({ this->m_out_size }, 0);
			m_W = W;
			m_b = b;
		}

	public:
		void forward(const TensorD& pre_data) override
		{
			m_z = pre_data.matmul(m_W) + m_b;
			m_a = m_activeFunc->activate(m_z);
		}

		void backprop(const TensorD& pre_data, const TensorD& for_data) override
		{
			TensorD dLz = m_activeFunc->apply_jacobian(m_z, m_a, for_data);
			int col = pre_data.shape().back();
			m_dw = pre_data.transpose().matmul(dLz) / col;
			m_db = dLz.mean();
			m_din = m_W.matmul(dLz.transpose()).transpose();
		}

		void update(Optimizer& opt) override
		{
			opt.update(m_dw, m_W);
			opt.update(m_db, m_b);
		}

		TensorD& output() override
		{
			return m_a;
		}

		TensorD& back_data() override
		{
			return m_din;
		}

	};
}
#endif !EINSNN_FULLY_CONNECTED_H_