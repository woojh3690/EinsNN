#ifndef EINSNN_ADAM_OPTIMIZER_H_
#define EINSNN_ADAM_OPTIMIZER_H_

#include "Optimizer.h"
#include <Tensor.h>
#include <unordered_map>

namespace EinsNN
{
	class AdamOptimizer : public Optimizer
	{
	private:
		double m_epsilon;
		double m_beta1, m_beta2;
		double m_beta1t, m_beta2t;
		std::unordered_map<const TensorD*, TensorD> m_hist_m, m_hist_v;

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

	private:
		void init(double beta1, double beta2, double epsilon)
		{
			m_beta1 = beta1;
			m_beta2 = beta2;
			m_beta1t = beta1;
			m_beta2t = beta2;
			m_epsilon = epsilon;
		}

	public:
		void update(TensorD& dvec, TensorD& tsr) override
		{
			TensorD* temp_dvec = &dvec;
			TensorD* temp_tsr = &tsr;
			TensorD& mvec = m_hist_m[&dvec];
			TensorD& vvec = m_hist_v[&dvec];

			if (mvec.shape() == vector<int>())
			{
				mvec.resize(dvec.shape());
				mvec.fill(1);
			}

			if (vvec.shape() == vector<int>())
			{
				vvec.resize(dvec.shape());
				vvec.fill(1);
			}

			mvec = m_beta1 * mvec + (1 - m_beta1) * dvec;
			vvec = m_beta2 * vvec + (1 - m_beta2) * dvec.pow();

			TensorD mvec_hat = mvec / (1 - m_beta1t);
			TensorD vvec_hat = vvec / (1 - m_beta2t);

			tsr = tsr - (m_lrate * mvec_hat / (vvec_hat.sqrt() + m_epsilon));

			m_beta1t = m_beta1t * m_beta1;
			m_beta2t = m_beta2t * m_beta2;
		}
	};
}
#endif // !EINSNN_ADAM_OPTIMIZER_H_
