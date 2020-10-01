#include<bits/stdc++.h>
using namespace std;
int main()
{
	int n;
	cin>>n;
	while(n--)
	{
		string str;
		cin>>str;
		int num = str.length();
		if(num <= 10)
		 cout<< str << endl;
		else 
		{
			cout<<str[0];
		    cout<<num-2;
		    cout<<str[num-1]<<endl;
		}
	}
	return 0;
}
