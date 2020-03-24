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
		TensorD& activate(const TensorD& Z) override
		{
			Tensor<bool> boolTsr = (Z > 0);
			TensorD exp_minuse = (Z.exp() - 1);
			TensorD right = exp_minuse * 0.01;
			TensorD* a = &boolTsr.select(Z, right);
			return *a;
		}

		TensorD& apply_jacobian(const TensorD& Z, const TensorD& A, 
			const TensorD& F) override
		{
			/*G.array() = (A.array() > Scalar(0)).select(
				F, (A.array() + Scalar(0.01)) * F.array()
			);*/
			return (A > 0).select(F, (A + 0.01) * F);
		}
	};
}
#endif // !_EINSNN_ELU_H_
