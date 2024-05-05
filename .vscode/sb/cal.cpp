#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <locale>
/**
 * @brief This function reads a text file and counts the number of messages sent by each user on different dates.
 *
 * @return int Returns 0 if the function executes successfully, otherwise returns 1.
 */
int main()
{
    // Open the file
    std::ifstream file("ssb.txt");
    if (!file)
    {
        std::cout << "Failed to open the file." << std::endl;
        return 1;
    }
    // Initialize a map to store the user statistics
    std::map<std::string, std::map<std::string, int>> userStats;
    // Declare variables for line, timestamp, and username
    std::string line, timestamp, username;
    // Read each line from the file
    while (std::getline(file, line))
    {
        if (!line.empty())
        { // Ignore empty lines
            if (line.substr(0, 10).find("2023-0") != std::string::npos)
            {                                   // Check if the line contains a timestamp
                timestamp = line.substr(0, 10); // Extract the date from the line
                // Find the position of the last space character
                size_t usernameStart = line.find_last_of(' ') + 1;
                // Extract the username from the line
                username = line.substr(usernameStart);
            }
            else
            {
                // Increment the count for the specific date for the user
                userStats[username][timestamp]++;
            }
        }
    }
    // Print the user statistics
    std::cout << "Username\tMonth.Day\tCount" << std::endl;
    for (const auto &user : userStats)
    {
        for (const auto &date : user.second)
        {
            std::cout << user.first << "\t" << date.first << "\t" << date.second << std::endl;
        }
    }
    return 0;
}