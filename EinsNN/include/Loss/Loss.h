#ifndef _EINSNN_LOSSLAYER_H_
#define _EINSNN_LOSSLAYER_H_

#include <string>
#include "../Config.h"

namespace EinsNN
{
	class Loss
	{
	protected:
		TensorD m_din; // 레이어의 도함수
		std::string m_type;

	public:
		Loss(std::string type) : m_type(type){}
		virtual ~Loss() {}

	public:
		std::string get_type()
		{
			return m_type;
		}

	public:
		virtual void evaluate(const TensorD& y_hat, const TensorD& target) = 0;
		virtual TensorD loss() = 0;
		virtual TensorD& back_data() = 0;
	};

}
#endif // !_EINSNN_LOSSLAYER_H_

