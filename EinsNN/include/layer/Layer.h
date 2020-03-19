#pragma once
#ifndef EINSNN_LAYER_H_
#define EINSNN_LAYER_H_

#include "../Config.h"

namespace EinsNN
{
	class Layer
	{
	protected:
		int m_in_size; // 레이어 입력 데이터 사이즈
		int m_out_size; // 레이어 출력 데이터 사이즈

	public:
		Layer(const int in_size, const int out_size) :
			m_in_size(in_size), m_out_size(out_size)
		{}
		~Layer() {};

		virtual void init() = 0;

	public:
		/*
		* @brief 순전파
		* @param pre_data 이전 레이어에서 활성화 함수를 통과한 데이터들이다.
		*/
		virtual void forward(TensorD pre_data) = 0;

		/*
		* @brief 역전파
		* @param pre_data 이전 레이어에서 활성화 함수를 통과한 데이터들이다.
		* @param for_data 다음 레이어에서 계산된 미분값들이다.
		*/
		virtual void backprop(TensorD pre_data, 
			const Tensor<double> for_data) = 0;

		virtual TensorD& output() = 0;

		virtual TensorD& get_din() = 0;

	private:
	};
} //namespace EinsNN

#endif // !EINSNN_LAYER_H_