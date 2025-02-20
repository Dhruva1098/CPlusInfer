//
// Created by Jyoti Singh on 17/02/25.
//
#include <fstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>

using json = nlohmann::json;

int main()
{
    std::ifstream f("model_info.json");
    json data = json::parse(f);
    std::cout << data.dump() << std::endl;

    std::ifstream ifs("model_info.json");
    json jf = json::parse(ifs);

    std::string str(R"({"json": "beta"})");
    json js = json::parse(str);
    std::cout << js.dump() << std::endl;

    std::cout << std::setw(4) << json::meta() << std::endl;
}