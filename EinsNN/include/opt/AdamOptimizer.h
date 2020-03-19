#ifndef EINSNN_ADAM_OPTIMIZER_H_
#define EINSNN_ADAM_OPTIMIZER_H_

#include "Optimizer.h"
#include <Tensor.h>

namespace EinsNN
{
	class AdamOptimizer : public Optimizer
	{
	private:
		double m_beta1;
		double m_beta2;
		double m_epsilon;

	public:
		AdamOptimizer(
			const double learning_rate = 0.001, 
			const double beta1 = 0.9, 
			const double beta2 = 0.999, 
			const double epsilon = 1e-7, 
			const string name = "Adam") :
			Optimizer(learning_rate, name)
		{
			init(beta1, beta2, epsilon);
		}
		~AdamOptimizer() {};

	public:
		/*void set_Learning_Rate(double learning_rate)
		{
			m_learning_rate = learning_rate;
		}*/
	private:
		void init(double beta1, double beta2, double epsilon)
		{
			m_beta1 = beta1;
			m_beta2 = beta2;
			m_epsilon = epsilon;
		}
	};
}
#endif // !EINSNN_ADAM_OPTIMIZER_H_
