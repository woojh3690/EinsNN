#ifndef _EINSNN_RELU_H_
#define _EINSNN_RELU_H_ "ReLU"

#include <string>
#include "../Config.h"
#include "Activation.h"

namespace EinsNN
{
	class ReLU : public Activation
	{
	public:
		ReLU() : Activation(_EINSNN_RELU_H_)
		{}

	public:
		TensorD activate(const TensorD& Z) override
		{
			TensorD zero(Z.shape(), 0);
			return (Z > 0).select(Z, zero);
		}

		TensorD apply_jacobian(const TensorD& Z, const TensorD& A,
			const TensorD& F) override
		{
			TensorD zero(Z.shape(), 0);
			return (A > 0).select(F, zero);
		}
	};
}
#endif // !_EINSNN_RELU_H_
