#ifndef EINSNN_FULLY_CONNECTED_H_
#define EINSNN_FULLY_CONNECTED_H_ "Fully"

#include "Layer.h"
#include "../Config.h"
#include "../activation/Activation.h"
#include "../activation/NoneActivation.h"

namespace EinsNN
{
	class Fully_connected : public Layer
	{
	private:
		TensorD m_W, m_dw;
		TensorD m_b, m_db;
		TensorD m_z;
		TensorD m_a;

		Activation* m_activeFunc = nullptr;

	public:
		Fully_connected(const int input_size, const int output_size) :
			Layer(input_size, output_size)
		{
			m_activeFunc = new NoneActivation();
		}

		Fully_connected(const int input_size, const int output_size, Activation* activeFunc) :
			Layer(input_size, output_size), m_activeFunc(activeFunc)
		{}
		~Fully_connected()
		{
			delete m_activeFunc;
		}

		void init() override
		{
			TensorD W({ this->m_in_size, this->m_out_size });
			TensorD b({ this->m_out_size }, 0);
			m_W = W.randomInit(-5, 5) / sqrt(m_in_size / 2);
			m_b = b;
		}

	public:
		void forward(const TensorD& pre_data) override
		{
			m_z = pre_data.matmul(m_W) + m_b;
			m_a = m_activeFunc->activate(m_z);
		}

		void backprop(const TensorD& pre_data, const TensorD& for_data) override
		{
			TensorD dLz = m_activeFunc->apply_jacobian(m_z, m_a, for_data);
			int col = pre_data.shape().back();
			m_dw = pre_data.transpose().matmul(dLz) / col;
			m_db = dLz.mean();
			m_din = m_W.matmul(dLz.transpose()).transpose();
		}

		void update(Optimizer& opt) override
		{
			opt.update(m_dw, m_W);
			opt.update(m_db, m_b);
		}

		TensorD output() override
		{
			return m_a;
		}

		TensorD back_data() override
		{
			return m_din;
		}

		string get_type() override
		{
			return EINSNN_FULLY_CONNECTED_H_;
		}

		vector<string> get_hiper_param() override
		{
			vector<string> param;
			param.push_back(to_string(m_in_size));
			param.push_back(to_string(m_out_size));
			param.push_back(m_activeFunc->return_type());
			return param;
		}

		string get_weight() override
		{
			return m_W.toString() + m_b.toString();
		}

		void set_weight(const string& weights) override
		{
			string::size_type idx = weights.find("][");
			if (idx == string::npos)
				throw invalid_argument("Save file is broken.");

			string strTsr_W = weights.substr(0, idx + 1);
			string strTsr_b = weights.substr(idx + 1, weights.size() - idx + 1);

			m_W.loadFromString(strTsr_W);
			m_b.loadFromString(strTsr_b);
		}

	};
}
#endif !EINSNN_FULLY_CONNECTED_H_