#ifndef _EINSNN_LOSSLAYER_H_
#define _EINSNN_LOSSLAYER_H_

#include "../Config.h"

namespace EinsNN
{
	class Loss
	{
	protected:
		TensorD m_din; // 레이어의 도함수

	public:
		Loss() {}
		virtual ~Loss() {}

	public:
		virtual void evaluate(const TensorD& y_hat, const TensorD& target) = 0;
		virtual TensorD loss() = 0;
		virtual TensorD& back_data() = 0;
	};

}
#endif // !_EINSNN_LOSSLAYER_H_

