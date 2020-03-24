#ifndef _EINSNN_CALLBACK_H_
#define _EINSNN_CALLBACK_H_

#include "../Config.h"
#include "../EinsNN.h"

namespace EinsNN
{
	class Model;

	class Callback
	{
	public:
		int epoche;

		Callback() : epoche(0)
		{
		}

		virtual ~Callback() {}

		virtual void pre_traning(EinsNN::Model* model, int epoche, TensorD x, TensorD y) = 0;
		virtual void post_traning(EinsNN::Model* model, int epoche, TensorD x, TensorD y) = 0;
	};
}
#endif // !_EINSNN_CALLBACK_H_
