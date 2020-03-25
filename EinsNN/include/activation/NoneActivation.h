#ifndef _EINSNN_NONEACTIVATION_H_
#define _EINSNN_NONEACTIVATION_H_

#include "Activation.h"

namespace EinsNN
{
	class NoneActivation : public Activation
	{
	public:
		NoneActivation() : Activation("None")
		{}
		~NoneActivation() {};

	public:
		TensorD activate(const TensorD& Z) override 
		{
			TensorD* newZ = new TensorD(Z);
			return *newZ;
		}

		TensorD apply_jacobian(const TensorD& Z, const TensorD& A, 
			const TensorD& F) override
		{
			TensorD* newF = new TensorD(F);
			return *newF;
		}

	};
}
#endif // ! _EINSNN_NONEACTIVATION_H_

