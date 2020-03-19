#ifndef _EINSNN_H_
#define _EINSNN_H_

#include <vector>
#include <string>
#include "Config.h"
#include "layer/Layer.h"
#include "opt/Optimizer.h"
#include "Loss/Loss.h"

namespace EinsNN
{

	class Model
	{
	public:
		Model() {};
		~Model() {};

	public:
		void set_layer(Layer* layer)
		{
			m_layers.push_back(layer);
		}

		Layer* get_layer(int layer_index)
		{
			return m_layers[layer_index];
		}

		void init()
		{
			for (int i = 0; i < m_layers.size(); i++)
			{
				m_layers[i]->init();
			}
		}

		void fit(TensorD& x, TensorD& y, int epoche)
		{
			init();

			for (int i = 0; i < epoche; i++)
			{
				this->forward(x);
				this->backprop(x, y);
				this->update();
			}
		}

		void compile(Loss& loss, Optimizer& opt)
		{
			m_loss = &loss;
			m_opt = &opt;
		}


		TensorD predict(TensorD& x)
		{
			if (m_layers.size() <= 0)
			{
				return TensorD();
			}

			this->forward(x);
			return m_layers.back()->output();
		}

		string preview()
		{
			return "";
		}

	private:
		vector<Layer*> m_layers;
		Optimizer* m_opt;
		Loss* m_loss;

		void forward(TensorD x)
		{
			TensorD output = x;
			for (auto layer : m_layers)
			{
				layer->forward(output);
				output = layer->output();
			}
		}

		void backprop(TensorD x, TensorD y)
		{
			// 마지막 히든레이어 역전파 계산
			TensorD back_data = y;
			for (size_t i = m_layers.size() - 1; i > 0; i--)
			{
				m_layers[i]->backprop(m_layers[i - 1]->output(), back_data);
				back_data = m_layers[i]->back_data();
			}

			// 첫번째 히든레이어 역전파 계산
			m_layers.front()->backprop(x, m_layers[1]->back_data());
		}

		void update()
		{

		}
	};
}

#endif // !EINSNN_H_