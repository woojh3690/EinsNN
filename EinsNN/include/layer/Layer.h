#pragma once
#ifndef EINSNN_LAYER_H_
#define EINSNN_LAYER_H_

#include "../Config.h"
#include "../opt/Optimizer.h"
#include <string>
#include <vector>

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
		virtual ~Layer() {}

		int in_size() const
		{
			return m_in_size;
		}

		int out_size() const
		{
			return m_out_size;
		}

	public:
		virtual void init() = 0;

		/*
		* @brief 순전파
		* @param pre_data 이전 레이어에서 활성화 함수를 통과한 데이터들이다.
		*/
		virtual void forward(const TensorD& pre_data) = 0;

		/*
		* @brief 역전파
		* @param pre_data 이전 레이어에서 활성화 함수를 통과한 데이터들이다.
		* @param for_data 다음 레이어에서 계산된 미분값들이다.
		*/
		virtual void backprop(const TensorD& pre_data, 
			const TensorD& for_data) = 0;

		/*
		* @brief 업데이트
		* @param opt 최적화 모듈
		*/
		virtual void update(Optimizer& opt) = 0;

		virtual TensorD output() = 0;

		virtual TensorD back_data() = 0;

		virtual string get_name() = 0;

		virtual vector<string> get_hiper_param() = 0;

		virtual string get_weight() = 0;

	};
} //namespace EinsNN

#endif // !EINSNN_LAYER_H_