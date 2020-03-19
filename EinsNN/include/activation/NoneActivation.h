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
		TensorD& activate(TensorD& Z) override 
		{
			return Z;
		}

		TensorD& apply_jacobian(TensorD& Z, TensorD& A, TensorD& F) override
		{
			return F;
		}

	};
}
#endif // ! _EINSNN_NONEACTIVATION_H_

