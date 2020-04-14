#ifndef _EINSNN_BATCHQUEUE_H_
#define _EINSNN_BATCHQUEUE_H_

#include "Config.h"
#include <time.h>
#include <algorithm>

namespace EinsNN
{
	class BatchQueue
	{
	private:
		TensorD shuffle_x;
		TensorD shuffle_y;

		int m_cursor = 0;
		int m_size = 0;

	public:
		BatchQueue()
		{}

		~BatchQueue()
		{}

		void inQueue(const TensorD& x, const TensorD& y, const int seed = time(0))
		{
			// 길이 검사
			if (x.size() != y.size())
			{
				throw invalid_argument("x and y have to same shape.");
			}

			m_size = x.size();

			shuffle_x = x;
			shuffle_y = y;

			std::srand(seed);
#pragma omp parallel for
			for (int i = 0; i < m_size; i++)
			{
				change(
					shuffle_x, shuffle_y, 
					random_num(0, m_size - 1), 
					random_num(0, m_size - 1)
				);
			}
		}

		bool next(TensorD* x, TensorD* y, const int batch_size)
		{
			if (m_cursor >= m_size)
			{
				return false;
			}

			bool exist_next = true;
			x->clear();
			y->clear();

			for (int i = 0; i < batch_size; i++)
			{
				x->append(shuffle_x[m_cursor]);
				y->append(shuffle_y[m_cursor]);
				m_cursor++;
				if (m_cursor >= m_size)
				{
					break;
				}
			}

			return true;
		}

		void move_cursor_front()
		{
			m_cursor = 0;
		}

	private:
		int random_num(int min, int max)
		{
			return rand() % (max - min + 1) + min;
		}

		void change(TensorD& x, TensorD& y, int left, int right)
		{
			TensorD temp;
			temp = x[left];
			x[left] = x[right];
			x[right] = temp;

			temp = y[left];
			y[left] = y[right];
			y[right] = temp;
		}
	};
}
#endif // !_EINSNN_BATCHQUEUE_H_
