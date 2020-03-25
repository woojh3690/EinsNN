#ifndef _EINSNN_ELU_H_
#define _EINSNN_ELU_H_

#include <string>
#include "../Config.h"
#include "Activation.h"

namespace EinsNN
{
	class ELU : public Activation
	{
	public:
		ELU() : Activation("ELU")
		{}
		~ELU() {}

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
