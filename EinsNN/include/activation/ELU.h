#ifndef _EINSNN_ELU_H_
#define _EINSNN_ELU_H_

#include <string>
#include "../Config.h"
#include "Activation.h"

namespace EinsNN
{
	class ELU : public Activation
	{
	private:
		TensorD m_zero;
		TensorD m_one;
		TensorD m_001;

	public:
		ELU() : Activation("ELU")
		{
			m_zero.append(0);
			m_one.append(1);
			m_001.append(0.01);
		};
		~ELU() {};

	public:
		TensorD& activate(TensorD& Z) override
		{
			Tensor<bool> boolTsr = (Z > m_zero);
			TensorD exp_minuse = (Z.exp() - m_one);
			TensorD right = exp_minuse * m_001;
			TensorD* a = &boolTsr.select(Z, right);
			return *a;
		}

		TensorD& apply_jacobian(TensorD& Z, TensorD& A, TensorD& F) override
		{
			/*G.array() = (A.array() > Scalar(0)).select(
				F, (A.array() + Scalar(0.01)) * F.array()
			);*/
			return (A > m_zero).select(F, (A + m_001) * F);
		}
	};
}
#endif // !_EINSNN_ELU_H_
