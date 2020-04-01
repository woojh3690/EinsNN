#include <iostream>
#include "include/EinsNN.h"
#include "include/layer/Fully_Connected.h"
#include "include/opt/AdamOptimizer.h"
#include "include/activation/ELU.h"
#include "include/Loss/MSE.h"
#include "include/Callback/VerboseCallback.h"

using namespace EinsNN;

int main()
{
	Model model;
	model.set_layer(new Fully_connected(2, 2, new ELU()));
	model.set_layer(new Fully_connected(2, 1));

	AdamOptimizer adam(0.02);
	MSE mse;
	model.compile(mse, adam);

	TensorD x({ 4, 2 });
	x[0][0] = 1;
	x[0][1] = 2;
	x[1][0] = 3;
	x[1][1] = 4;
	x[2][0] = 5;
	x[2][1] = 6;
	x[3][0] = 7;
	x[3][1] = 8;

	vector<int> a = { 4, 1 };
	TensorD y(a);
	y[0][0] = 4;
	y[1][0] = 10;
	y[2][0] = 16;
	y[3][0] = 22;

	VerboseCallback callback;
	model.fit(x, y, 4, 50, callback);

	// 학습된 모델 평가
	Matrix::Tensor<double> y_pred = model.predict(x);

	MSE new_mse;
	new_mse.evaluate(y_pred, y);
	double eval_loss = new_mse.loss().value();
	string path = "./save-1.txt";
	model.save(path);
	model.load(path);
	return 0;
}
