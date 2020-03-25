// EinsNN.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

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
	model.set_layer(new Fully_connected(2, 1, new ELU()));
	model.set_layer(new Fully_connected(1, 1, new ELU()));
	model.set_layer(new Fully_connected(1, 1));

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
	model.fit(x, y, 100, 6000, callback);
	Tensor<double> temp = model.predict(x);

	return 0;
}
