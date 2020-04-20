#ifndef _EINSNN_VERBOSECALLBACK_H_
#define _EINSNN_VERBOSECALLBACK_H_

#include "callback.h"
#include "../einsnn.h"
#include <iostream>

namespace EinsNN
{

	class VerboseCallback : public Callback
	{
		void pre_traning(int epoche, const TensorD& x, const TensorD& y) override
		{

		}

		void post_traning(const TensorD loss, int epoche, 
			const TensorD& x, const TensorD& y) override
		{
			char strPrint[100];
			char strFormat[100] = "[Epoch %d] Loss : %lf";
			std::snprintf(strPrint, 100, strFormat, epoche, loss[0].value());
			std::cout << strPrint << std::endl;
		}
	};
}
#endif // !VerboseCallback
