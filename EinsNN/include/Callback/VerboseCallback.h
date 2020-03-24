#ifndef _EINSNN_VERBOSECALLBACK_H_
#define _EINSNN_VERBOSECALLBACK_H_

#include "Callback.h"
#include "../EinsNN.h"
#include <iostream>

namespace EinsNN
{
	class VerboseCallback : public EinsNN::Callback
	{
		void pre_traning(EinsNN::Model* model, int epoche, TensorD x, TensorD y) override
		{

		}

		void post_traning(EinsNN::Model* model, int epoche, TensorD x, TensorD y) override
		{
			TensorD loss = model->get_loss().loss();
			char strPrint[100];
			char strFormat[100] = "[Epoch %d] Loss : %lf";
			std::sprintf(strPrint, strFormat, epoche, loss[0].value());
			std::cout << strPrint << std::endl;
		}
	};
}
#endif // !VerboseCallback
