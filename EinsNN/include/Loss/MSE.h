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

		virtual void evaluate(const TensorD& y_hat, const TensorD& target) override
		{
			checkData(y_hat, target);
			m_din = y_hat - target;
		}

		TensorD loss() override
		{
			return (m_din.pow().mean())[0];
		}

		TensorD& back_data() override
		{
			return m_din;
		}

	private:
		void checkData(const TensorD& y_hat, const TensorD& target)
		{
			vector<int> y_hat_shape = y_hat.shape();
			vector<int> target_shape = target.shape();

			if (y_hat_shape.size() != 2 || target_shape.size() != 2)
			{
				throw invalid_argument("2차원만 지원");
			}

			if (y_hat_shape != target_shape)
			{
				throw invalid_argument("shape이 동일하지 않습니다.");
			}
		}
	};
}
#endif // !_EINSNN_MSE_H_

