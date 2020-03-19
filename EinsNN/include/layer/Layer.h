#pragma once
#ifndef EINSNN_LAYER_H_
#define EINSNN_LAYER_H_

#include "../Config.h"

namespace EinsNN
{
	class Layer
	{
	protected:
		// i/o 정의 -1이라면 자유
		int m_in_size; // 레이어 입력 데이터 사이즈.
		int m_out_size; // 레이어 출력 데이터 사이즈
		TensorD m_din; // 레이어의 도함수

	public:
		Layer(const int in_size, const int out_size) :
			m_in_size(in_size), m_out_size(out_size)
		{}
		~Layer() {}

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
		virtual void backprop(TensorD pre_data, TensorD for_data) = 0;

		virtual TensorD& output() = 0;

		virtual TensorD& back_data() = 0;

	};
} //namespace EinsNN

#endif // !EINSNN_LAYER_H_