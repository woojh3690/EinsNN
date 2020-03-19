#ifndef EINSNN_OPTIMIZER_H_
#define EINSNN_OPTIMIZER_H_

#include <Tensor.h>
#include <string>

namespace EinsNN
{
	class Optimizer
	{
	protected:
		double m_learning_rate;
		string m_name;

	public:
		Optimizer(const double learning_rate, const string name):
			m_learning_rate(learning_rate), m_name(name)
		{}
		~Optimizer() {};

	public:
		void set_Learning_Rate(double learning_rate)
		{
			m_learning_rate = learning_rate;
		}
	};
}

#endif // !EINSNN_OPTIMIZER_H_

