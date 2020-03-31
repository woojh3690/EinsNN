#ifndef _EINSNN_MODEL_H_
#define _EINSNN_MODEL_H_

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include "Config.h"
#include "layer/Layer.h"
#include "opt/Optimizer.h"
#include "Loss/Loss.h"
#include "Callback/Callback.h"
#include "BatchQueue.h"
#include "layer/Fully_Connected.h"
#include "activation/ELU.h"

std::string ReplaceAll(std::string &str, const std::string& from, const std::string& to) {
	size_t start_pos = 0; //stringó������ �˻�
	while ((start_pos = str.find(from, start_pos)) != std::string::npos)  //from�� ã�� �� ���� ������
	{
		str.replace(start_pos, from.length(), to);
		start_pos += to.length(); // �ߺ��˻縦 ���ϰ� from.length() > to.length()�� ��츦 ���ؼ�
	}
	return str;
}

vector<string> tokenize_getline(const string& data, const char delimiter = ' ') {
	vector<string> result;
	string token;
	stringstream ss(data);

	while (getline(ss, token, delimiter)) {
		result.push_back(token);
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

		void fit(const TensorD& x, const TensorD& y, const int batch_size, 
			int epoche, Callback& callback)
		{
			init();

			// ť
			BatchQueue queue;
			queue.inQueue(x, y);

			// �н�
			TensorD batch_x;
			TensorD batch_y;
			double global_loss;
			int batch_count;
			for (int i = 0; i < epoche; i++)
			{
				callback.pre_traning(i, batch_x, batch_y);

				// ��ġ �н�
				global_loss = 0;
				batch_count = 0;
				queue.move_cursor_front();
				while (queue.next(&batch_x, &batch_y, batch_size))
				{
					this->forward(batch_x);
					this->backprop(batch_x, batch_y);
					this->update();
					global_loss += m_loss->loss().value();
					batch_count++;
				}
				callback.post_traning(global_loss / batch_count, i, batch_x, batch_y);
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
			if (stream.is_open())
			{
				for (auto layer : m_layers)
				{
					string name = layer->get_name();
					vector<string> hiper = layer->get_hiper_param();
					string weights = layer->get_weight();

					ReplaceAll(weights, "\n", "\\n");

					string strHiper;
					for (auto item : hiper)
					{
						strHiper += item + ", ";
					}
					strHiper.resize(strHiper.size() - 2);
					string strSave = "Layer=" + name + "(" + strHiper + ")" + 
						"{" + weights + "}";

					stream << strSave << endl;
				}
			}
			stream.close();
		}

		/**
		* @brief ����� ���Ͽ��� ���� �о�´�.
		*/
		void load(string path)
		{
			this->~Model();
			ifstream stream(path);
			if (stream.is_open())
			{
				string line;
				while (getline(stream, line))    //���� ������ �о����� Ȯ��
				{
					string name, weights;
					vector<string> hiper;
					layer_parser(line, name, hiper, weights);

					if (name == "Fully")
					{
						Fully_connected* fully = new Fully_connected(atoi(hiper[0].c_str()), 
							atoi(hiper[1].c_str()), new ELU());
						m_layers.push_back(fully);
					}


				}



			}
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

		bool layer_parser(const string& strLayer, string& name, 
			vector<string>& hipers, string& weights)
		{
			vector<string> one = tokenize_getline(strLayer, '=');
			
			string::size_type split_l = one[1].find("(");
			string::size_type split_r = one[1].find(")");

			name = one[1].substr(0, split_l);
			string strHipers = one[1].substr(split_l + 1, split_r - split_l - 1);

			hipers = tokenize_getline(strHipers, ',');

			split_l = one[1].find("{");
			split_r = one[1].find("}");
			weights = one[1].substr(split_l + 1, split_r - 1);
			return true;
		}
	};
}

#endif // !_EINSNN_MODEL_H_
