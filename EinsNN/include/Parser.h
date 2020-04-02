#ifndef _EINSNN_PARSER_H_
#define _EINSNN_PARSER_H_

#define TRIM_SPACE " \t\n"
#include <string>
#include <vector>
#include <sstream>

namespace EinsNN
{

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

	static std::string ReplaceAll(std::string &str, const std::string& from, const std::string& to) {
		size_t start_pos = 0; //stringó������ �˻�
		while ((start_pos = str.find(from, start_pos)) != std::string::npos)  //from�� ã�� �� ���� ������
		{
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // �ߺ��˻縦 ���ϰ� from.length() > to.length()�� ��츦 ���ؼ�
		}
		return str;
	}

	static std::vector<std::string> tokenize_getline(const std::string& data, const char delimiter = ' ') {
		std::vector<std::string> result;
		std::string token;
		std::stringstream ss(data);

		while (getline(ss, token, delimiter)) {
			result.push_back(trim(token));
		}
		return result;
	}
}

#endif // !_EINSNN_PARSER_H_

