#include<bits/stdc++.h>
using namespace std;
int main()
{
	string str;
	cin>>str;
	string res = "";
	for(char c : str)
	{
		if(c >= 'A' && c<='Z')
		   c += 'a' - 'A';
		  if (c != 'a' && c != 'e' && c != 'i' && c != 'o' && c != 'u' && c != 'y')
        {
            //res.append(1, '.');
            //res.append(1, c);
            res+= '.';
            res+= c;
        }
	}
  cout<<res<<endl;
	return 0;
}
