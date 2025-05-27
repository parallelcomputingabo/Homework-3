#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <string>

// This runs the main.cu through every folder so it's a bit easier to all results

namespace fs = std::filesystem;

int main() {
    std::string base_path = "data";

    for (int i = 0; i <= 9; ++i) {
        std::string folder = base_path + "/" + std::to_string(i);
        std::string input0 = folder + "/input0.raw";
        std::string input1 = folder + "/input1.raw";
        std::string reference = folder + "/output.raw";

        std::cout << "Running folder: " << folder << std::endl;
        std::string cmd = "./app 512 512 512 " + input0 + " " + input1 + " " + reference;
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "Failed to run: " << cmd << std::endl;
        }
    }

    return 0;
}