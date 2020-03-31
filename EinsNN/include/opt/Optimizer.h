#ifndef EINSNN_OPTIMIZER_H_
#define EINSNN_OPTIMIZER_H_

#include "../Config.h"
#include <string>

namespace EinsNN
{
	class Optimizer
	{
	protected:
		double m_lrate;
		string m_name;

	public:
		Optimizer(const double learning_rate, const string name):
			m_lrate(learning_rate), m_name(name)
		{}
		virtual ~Optimizer() {}

	public:
		void set_Learning_Rate(double learning_rate)
		{
			m_lrate = learning_rate;
		}

		virtual void update(TensorD& derivative, TensorD& tenosr) = 0;
	};
}

#endif // !EINSNN_OPTIMIZER_H_

