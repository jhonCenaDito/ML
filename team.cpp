#include<bits/stdc++.h>
using namespace std;
int main()
{
	int n;
	cin>>n;
	int arr[n][3];
	for(int i = 0; i<n; i++)
	{
		for(int j=0; j<3;j++)
		{
			cin>>arr[i][j];
		}
	}
	int res = 0;
	for(int i = 0; i<n; i++)
	{
		int sum = 0;
		for(int j=0; j<3;j++)
		{
			if(arr[i][j] == 1) sum++;
		}
		if(sum >= 2) res++;
	}
	
	cout <<res <<endl;
	
	return 0;
}
	
