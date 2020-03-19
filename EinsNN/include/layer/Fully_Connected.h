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
		TensorD m_W, m_dW;
		TensorD m_b, m_db;
		TensorD m_a;
		TensorD m_din;

		Activation* m_activeFunc = nullptr;

	public:
		Fully_connected(const int input_size, const int output_size) :
			Layer(input_size, output_size)
		{
			m_activeFunc = new NoneActivation();
		};

		Fully_connected(const int input_size, const int output_size, Activation* activeFunc) :
			Layer(input_size, output_size), m_activeFunc(activeFunc)
		{};
		~Fully_connected();

		void init()
		{
			TensorD W({ this->m_in_size, this->m_out_size }, 1);
			TensorD b({ this->m_out_size }, 1);
			m_W = W;
			m_b = b;
		}

	public:
		void forward(TensorD pre_data) override
		{
			TensorD z = pre_data.matmul(m_W) + m_b;
			m_a = m_activeFunc->activate(z);
		}

		void backprop(TensorD pre_data, TensorD for_data) override
		{
			m_din;
		}

		TensorD& output() override
		{
			return m_a;
		}

		TensorD& get_din() override
		{
			return m_din;
		}

	};
}
#endif !EINSNN_FULLY_CONNECTED_H_