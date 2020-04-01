#ifndef _EINSNN_NONEACTIVATION_H_
#define _EINSNN_NONEACTIVATION_H_ "None"

#include "Activation.h"

namespace EinsNN
{
	class NoneActivation : public Activation
	{
	public:
		NoneActivation() : Activation(_EINSNN_NONEACTIVATION_H_)
		{}

	public:
		TensorD activate(const TensorD& Z) override 
		{
			TensorD* newZ = new TensorD(Z);
			return *newZ;
		}

		TensorD apply_jacobian(const TensorD& Z, const TensorD& A, 
			const TensorD& F) override
		{
			return F;
		}

	};
}
#endif // ! _EINSNN_NONEACTIVATION_H_

