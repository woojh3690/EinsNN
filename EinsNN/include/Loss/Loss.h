#ifndef _EINSNN_LOSSLAYER_H_
#define _EINSNN_LOSSLAYER_H_

#include "../Config.h"

namespace EinsNN
{
	class Loss
	{
	protected:
		TensorD m_din; // ���̾��� ���Լ�

	public:
		Loss() {}
		~Loss() {}

	public:
		virtual void evaluate(TensorD& y_hat, TensorD& target) = 0;
		virtual TensorD& loss() = 0;
		virtual TensorD& back_data() = 0;
	};

}
#endif // !_EINSNN_LOSSLAYER_H_

