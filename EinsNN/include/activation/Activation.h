#ifndef _EINSNN_ACTIVATION_H_
#define _EINSNN_ACTIVATION_H_

#include <string>
#include "../Config.h"

namespace EinsNN
{
	class Activation
	{
	protected:
		std::string name = "None";
	public:
		Activation() {};
		Activation(string name) : 
			name(name) {}
		~Activation() {};

	public:
		virtual TensorD activate(const TensorD& Z) = 0;
		//{
		//	TensorD* aVirtual = new TensorD();
		//	return *aVirtual;
		//};
		virtual TensorD apply_jacobian(const TensorD& Z, const TensorD& A, 
			const TensorD& F) = 0;
		//{
		//	TensorD* aVirtual = new TensorD();
		//	return *aVirtual;
		//};

		std::string return_type()
		{
			return name;
		}
	};
}
#endif // !_EINSNN_ACTIVATION_H_

