#ifndef _EINSNN_SELECTOR_H_
#define _EINSNN_SELECTOR_H_

#include "layer/Layer.h" 
#include "layer/Fully_Connected.h"

#include "activation/Activation.h"
#include "activation/NoneActivation.h"
#include "activation/ELU.h"
#include "activation/ReLU.h"

#include "Loss/Loss.h"
#include "Loss/MSE.h"

#include "opt/Optimizer.h"
#include "opt/AdamOptimizer.h"

#include "Callback/Callback.h"
#include "Callback/VerboseCallback.h"

namespace EinsNN
{
	class Selector
	{
	public:
		Selector()
		{}

		~Selector()
		{}

	public:
		static Layer* selectLayer(const string& type, const vector<string>& hipers, 
			const string& weights)
		{
			if (type == EINSNN_FULLY_CONNECTED_H_)
			{
				int in_size = stoi(hipers[0]);
				int out_size = stoi(hipers[1]);
				string actType = hipers[2];

				// 활성화 함수 선택
				Activation* act = nullptr;
				if (actType == _EINSNN_NONEACTIVATION_H_)
					act = new NoneActivation();
				else if (actType == _EINSNN_ELU_H_)
					act = new ELU();
				else if (actType == _EINSNN_RELU_H_)
					act = new ReLU();
				else
					throw invalid_argument("Save file is broken.");

				Fully_connected* fully = new Fully_connected(in_size, out_size, act);
				fully->set_weight(weights);
				return fully;
			}
		}

		static Loss* selectLoss(string type)
		{
			if (type == _EINSNN_MSE_H_)
			{
				return new MSE();
			}
		}

		static Optimizer* selectOpt(string type, vector<string> hipers)
		{
			if (type == _EINSNN_ADAM_OPTIMIZER_H_)
			{
				double beta1 = stod(hipers[0]);
				double beta2 = stod(hipers[1]);
				double epsilon = stod(hipers[2]);

				return new AdamOptimizer(beta1, beta2, epsilon);
			}
		}
	};
}
#endif // !_EINS_SELECTOR_H_

