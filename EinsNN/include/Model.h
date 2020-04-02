#ifndef _EINSNN_MODEL_H_
#define _EINSNN_MODEL_H_

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "Config.h"
#include "layer/Layer.h"
#include "Loss/Loss.h"
#include "Callback/Callback.h"
#include "opt/Optimizer.h"
#include "BatchQueue.h"
#include "Selector.h"

#define TRIM_SPACE " \t\n"

namespace ospace {
	inline std::string trim(std::string& s, const std::string& drop = TRIM_SPACE)
	{
		std::string r = s.erase(s.find_last_not_of(drop) + 1);
		return r.erase(0, r.find_first_not_of(drop));
	}
	inline std::string rtrim(std::string s, const std::string& drop = TRIM_SPACE)
	{
		return s.erase(s.find_last_not_of(drop) + 1);
	}
	inline std::string ltrim(std::string s, const std::string& drop = TRIM_SPACE)
	{
		return s.erase(0, s.find_first_not_of(drop));
	}

}
static std::string ReplaceAll(std::string &str, const std::string& from, const std::string& to) {
	size_t start_pos = 0; //stringó������ �˻�
	while ((start_pos = str.find(from, start_pos)) != std::string::npos)  //from�� ã�� �� ���� ������
	{
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // �ߺ��˻縦 ���ϰ� from.length() > to.length()�� ��츦 ���ؼ�
	}
	return str;
}

static vector<string> tokenize_getline(const string& data, const char delimiter = ' ') {
	vector<string> result;
	string token;
	stringstream ss(data);

	while (getline(ss, token, delimiter)) {
		result.push_back(ospace::trim(token));
	}
	return result;
}

namespace EinsNN
{
	class Model
	{
	public:
		Model() {}
		~Model() 
		{
			for (auto layer : m_layers)
			{
				delete layer;
			}


			if (m_opt != NULL)
			{
				if (!m_opt->get_type().empty())
				{
					delete m_opt;
				}
			}

			if (m_loss != NULL)
			{
				if (!m_loss->get_type().empty())
				{
					delete m_loss;
				}
			}
		}

	public:
		void set_layer(Layer* layer)
		{
			m_layers.push_back(layer);
			check_layer_io();
		}

		Layer* get_layer(int layer_index)
		{
			return m_layers[layer_index];
		}

		std::size_t num_layers()
		{
			return m_layers.size();
		}

		void init()
		{
			for (int i = 0; i < m_layers.size(); i++)
			{
				m_layers[i]->init();
			}
		}

		void fit(const TensorD& x, const TensorD& y, const int& batch_size,
			Callback& callback, const int& epoche, const double& scale = -1)
		{
			init();

			// ť
			BatchQueue queue;
			queue.inQueue(x, y);

			// �н�
			TensorD batch_x;
			TensorD batch_y;
			for (int i = 0; i < epoche; i++)
			{
				//callback
				callback.pre_traning(i, batch_x, batch_y);

				// ��ġ �н�
				queue.move_cursor_front();
				while (queue.next(&batch_x, &batch_y, batch_size))
				{
					this->forward(batch_x);
					this->backprop(batch_x, batch_y);
					this->update();
				}

				// �۷ι� �ν� ���
				this->forward(x);
				m_loss->evaluate(m_layers.back()->output(), y);
				double global_loss = m_loss->loss().value();

				// callback
				callback.post_traning(global_loss, i, batch_x, batch_y);

				if (global_loss < scale)
					break;
			}
		}

		void compile(Loss& loss, Optimizer& opt)
		{
			m_loss = &loss;
			m_opt = &opt;
		}

		TensorD predict(const TensorD& x)
		{
			if (m_layers.size() <= 0)
			{
				return *new TensorD();
			}

			this->forward(x);
			return m_layers.back()->output();
		}

		string preview() const
		{
			return "";
		}

		Loss& get_loss() const
		{
			return *m_loss;
		}

		/**
		* @brief �� ������ �Ķ���͵��� �����ϴ� �Լ�
		*/
		void save(string path)
		{
			ofstream stream(path);
			if (!stream.is_open())
				throw runtime_error("Can not open " + path);

			// layers
			for (auto layer : m_layers)
			{
				string type = layer->get_type();

				string strHiper;
				for (auto item : layer->get_hiper_param())
					strHiper += item + ", ";
				strHiper.resize(strHiper.size() - 2);

				string weights = layer->get_weight();
				ReplaceAll(weights, "\n", "\\n");

				string strSave = "Layer="+type+"("+strHiper+")"+"{"+weights+"}";
				stream << strSave << endl;
			}

			//loss
			stream << "Loss=" + m_loss->get_type() << endl;

			// opt
			string name = m_opt->get_type();
			string strHiper;
			for (auto item : m_opt->get_hiper_param())
				strHiper += item + ", ";
			strHiper.resize(strHiper.size() - 2);

			stream << "Opt=" + name + "(" + strHiper + ")" << endl;

			stream.close();
		}

		/**
		* @brief ����� ���Ͽ��� ���� �о�´�.
		*/
		void load(string path)
		{
			this->~Model();
			ifstream stream(path);
			if (!stream.is_open())
				throw runtime_error("Save file not found.");

			string line;
			while (getline(stream, line))    //���� ������ �о����� Ȯ��
			{
				string modelType;
				string type, weights;
				vector<string> hipers;
				line_parser(line, modelType, type, hipers, weights);

				if (modelType == "Layer")
				{
					Layer* layer = Selector::selectLayer(type, hipers, weights);
					m_layers.push_back(layer);
				}
				else if (modelType == "Loss")
				{
					m_loss = Selector::selectLoss(type);
				}
				else if (modelType == "Opt")
				{
					m_opt = Selector::selectOpt(type, hipers);
				}
			}
			stream.close();
		}

	private:
		vector<Layer*> m_layers;
		Optimizer* m_opt;
		Loss* m_loss;

		void forward(const TensorD& x)
		{
			TensorD output = x;
			for (auto layer : m_layers)
			{
				layer->forward(output);
				output = layer->output();
			}
		}

		void backprop(const TensorD& input, const TensorD& target)
		{
			// �����Լ� ���
			m_loss->evaluate(m_layers.back()->output(), target);
			TensorD back_data = m_loss->back_data();

			// ������ ���緹�̾� ������ ���
			for (size_t i = m_layers.size() - 1; i > 0; i--)
			{
				m_layers[i]->backprop(m_layers[i - 1]->output(), back_data);
				back_data = m_layers[i]->back_data();
			}

			// ù��° ���緹�̾� ������ ���
			m_layers.front()->backprop(input, m_layers[1]->back_data());
		}

		void update()
		{
			for (auto layer : m_layers)
			{
				layer->update(*m_opt);
			}
		}

		/**
		* @brief ���̾� ���� ����»����� �˻�
		*/
		void check_layer_io() const
		{
			if (m_layers.size() <= 1)
			{
				return;
			}

			for (int i = 1; i < m_layers.size(); i++)
			{
				if (m_layers[i]->in_size() != m_layers[i - 1]->out_size())
				{
					throw std::invalid_argument("Unit sizes does not match");
				}
			}
		}

		bool line_parser(string& strLayer, string& modelType, string& type,
			vector<string>& hipers, string& weights)
		{
			vector<string> token = tokenize_getline(strLayer, '=');
			modelType = token[0];
			
			string::size_type split_l = token[1].find("(");
			string::size_type split_r = token[1].find(")");

			type = token[1].substr(0, split_l);
			string strHipers = token[1].substr(split_l + 1, split_r - split_l - 1);

			hipers = tokenize_getline(strHipers, ',');

			split_l = token[1].find("{");
			split_r = token[1].find("}");
			weights = token[1].substr(split_l + 1, split_r - split_l-1);
			return true;
		}
	};
}

#endif // !_EINSNN_MODEL_H_
