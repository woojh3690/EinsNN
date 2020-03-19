#ifndef EINSNN_H_
#define EINSNN_H_

#include <vector>
#include <string>
#include "Config.h"
#include "layer/Layer.h"
#include "opt/Optimizer.h"
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

		void fit(TensorD& x, TensorD& y, int epoche, Optimizer& opt)
		{
			init();

			for (int i = 0; i < epoche; i++)
			{
				this->forward(x);
				this->backprop(x, y);
				this->update(opt);
			}
		}

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
			/*TensorD 
			for (auto layer : m_layers)
			{

			}*/
		}

		void update(Optimizer& opt)
		{

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
	};
}

#endif // !EINSNN_H_