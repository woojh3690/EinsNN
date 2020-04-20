#ifndef _EINSNN_ELU_H_
#define _EINSNN_ELU_H_ "ELU"

#include <string>
#include "../config.h"
#include "activation.h"

namespace EinsNN
{
	class ELU : public Activation
	{
	public:
		ELU() : Activation(_EINSNN_ELU_H_)
		{}

	public:
		TensorD activate(const TensorD& Z) override
		{
			return (Z > 0).select(Z, (Z.exp() - 1) * 0.01);
		}

		TensorD apply_jacobian(const TensorD& Z, const TensorD& A, 
			const TensorD& F) override
		{
			return (A > 0).select(F, (A + 0.01) * F);
		}
	};
}
#endif // !_EINSNN_ELU_H_
