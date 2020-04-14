#include <iostream>
#include "include/EinsNN.h"
#include "include/layer/Fully_Connected.h"
#include "include/opt/AdamOptimizer.h"
#include "include/activation/ELU.h"
#include "include/activation/ReLU.h"
#include "include/Loss/MSE.h"
#include "include/Callback/VerboseCallback.h"
#include <crtdbg.h>
using namespace EinsNN;

void memory_leak_test()
{
	Model model;
	model.set_layer(new Fully_connected(2, 2, new ReLU()));
	model.set_layer(new Fully_connected(2, 2, new ReLU()));
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
	model.fit(x, y, 4, callback, 1000);

	// 학습된 모델 평가
	TensorD y_pred = model.predict(x);

	string path = "./save-1.txt";

	// model 저장
	model.save(path);

	// model 로딩
	Model new_model;
	new_model.load(path);

	TensorD new_y_pred = new_model.predict(x);
}

int main()
{
	//_CrtSetBreakAlloc(3773);
	memory_leak_test();

	//_CrtDumpMemoryLeaks();
	return 0;
}
