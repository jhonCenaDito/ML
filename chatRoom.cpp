#include<bits/stdc++.h>
using namespace std;

int i,j = 0;
int main()
{
	string str;
	cin >> str;
	string test = "hello";
	// str = ahhellllloou

	
	while(str[i]!='\0' && test[j]!='\0')
	{
		if(j==5) break;
	  if(str[i] == test[j])
		j++;
	i++;
	}
	//cout<< j << endl;
	if(j == 5) cout<<"YES\n";
	else cout << "NO\n";
	
	return 0;
}
			
