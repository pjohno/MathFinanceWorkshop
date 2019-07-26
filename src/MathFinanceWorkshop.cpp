#include "MathFinanceWorkshop.hpp"
#include <iostream>
#include <cstdlib>
#include <string>
using namespace std;

namespace MathFinanceWorkshop
{
  int testProject(void)
  {
    string home_location=getenv("HOME");
    int exit_status = 0;
    cout << " My new project:: MathFinanceWorkshop " << endl;
    cout << " Home location " << home_location << endl;
    return exit_status;
  }
  
}

