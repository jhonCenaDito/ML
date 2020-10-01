#include<bits/stdc++.h>
using namespace std;
// (x/2)*y) --> gives the max number of domino to be placed in row 
// (x%2)*y)/2) --> computes the number of domino that can be accomodated in the rest of the cells
int main()
{
	int n,m;
	cin >> n >> m;
	int x = max(n,m);
	int y = min(n,m);
	int res = ((x/2)*y)+(((x%2)*y)/2);
	cout<<res<<endl;
	return 0;
}
