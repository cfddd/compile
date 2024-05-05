#include <bits/stdc++.h>
#define debug1(a) cout << #a << '=' << a << endl;
#define debug2(a, b) cout << #a << " = " << a << "  " << #b << " = " << b << endl;
#define debug3(a, b, c) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << endl;
#define debug4(a, b, c, d) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << endl;
#define debug5(a, b, c, d, e) cout << #a << " = " << a << "  " << #b << " = " << b << "  " << #c << " = " << c << "  " << #d << " = " << d << "  " << #e << " = " << e << endl;

using namespace std;
int main(){
	string s;
	cin >>s;
	map<char,int> cnt;
	long long ans = 0;
	int n= s.size();
	for(int i=0,j=0;i<n;i++){
		while(j<n&& s[j]!='d'){
			j++;
		}
		for(int l=i,r=i;l<j;l++){
			while(r<j&&(cnt['r']==0|| cnt['e']==0)){
				cnt[s[r]]++;
				r++;
			}
			if(cnt['r']&&cnt['e'])
			{
				ans += j-r+1;
				debug2(j,r);
			}
			cnt[s[l]]--;
		}
		i=j++;
	}
	cout << ans << endl;
}