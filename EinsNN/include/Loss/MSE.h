#ifndef _EINSNN_MSE_H_
#define _EINSNN_MSE_H_

#include "Loss.h"
namespace EinsNN
{
	class MSE : public Loss
	{
	public:
		MSE() : Loss() {}
		~MSE() {}

	public:

		virtual void evaluate(TensorD& y_hat, TensorD& target) override
		{
			checkData(y_hat, target);
			m_din = y_hat - target;
		}

		TensorD& loss() override
		{
			return this->m_din.pow().mean() * 0.5;
		}

		TensorD& back_data() override
		{
			return m_din;
		}

	private:
		void checkData(TensorD& y_hat, TensorD& target)
		{
			vector<int> y_hat_shape = y_hat.shape();
			vector<int> target_shape = target.shape();

			if (y_hat_shape.size() != 2 || target_shape.size() != 2)
			{
				throw invalid_argument("2차원만 지원");
			}

			if (y_hat_shape.front() != target_shape.front())
			{
				throw invalid_argument("row가 맞지 않습니다.");
			}
		}
	};
}
#endif // !_EINSNN_MSE_H_

